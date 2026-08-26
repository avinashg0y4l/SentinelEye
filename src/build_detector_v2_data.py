"""Build validated, sequence-aware SentinelEye detector-v2 datasets.

This script is deliberately non-destructive: it refuses to use an existing
output directory and never alters source datasets, V1 artifacts, or models.
It does not train a model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CSV_PATH = Path("data/raw/Multifire/data_structure.csv")
MULTIFIRE_TRAIN_ROOT = Path("data/experiments/detector_v2/multifire_manual_train/Train")
DFIRE_SPLIT_ROOT = Path("data/experiments/detector_v1/DFIRE_clean_split")

MULTIFIRE_OUTPUT = Path("data/experiments/detector_v2/multifire_manual_sequence_v2_safe")
COMBINED_OUTPUT = Path("data/experiments/detector_v2/combined_v2_safe")

# Verified from the official D-Fire aggregate class totals and local raw labels.
SMOKE_CLASS_ID = 0
FIRE_CLASS_ID = 1
CLASS_NAMES = {SMOKE_CLASS_ID: "Smoke", FIRE_CLASS_ID: "Fire"}

MIN_COMPONENT_PIXELS = 10
VAL_RATIO = 0.20
SEED = 42
VIDEO_PATTERN = re.compile(r"^FramesV(?P<video>\d+)_")
CATEGORY_FOLDER = {"urban": "FU", "rural": "FR"}


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Choose a new output directory or inspect the existing artifact."
        )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_yaml(path: Path, root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {root.resolve().as_posix()}",
                "",
                "train: train/images",
                "val: val/images",
                "",
                "names:",
                "  0: Smoke",
                "  1: Fire",
                "",
            ]
        ),
        encoding="utf-8",
    )


def video_id(image_name: str) -> str:
    match = VIDEO_PATTERN.match(image_name)
    if not match:
        raise ValueError(f"Cannot derive source-video identity from {image_name!r}")
    return f"V{match.group('video')}"


def split_by_video(records: list[dict]) -> dict[str, str]:
    """Assign whole videos to train/val while approximating 20% per category."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_video[record["source_video"]].append(record)

    category_totals = Counter(record["source_category"] for record in records)
    category_val_counts = Counter()
    targets = {category: total * VAL_RATIO for category, total in category_totals.items()}

    groups = list(by_video.items())
    random.Random(SEED).shuffle(groups)
    assignments: dict[str, str] = {}

    for group_id, group_records in groups:
        group_counts = Counter(record["source_category"] for record in group_records)
        current_error = sum(
            abs(category_val_counts[category] - targets[category])
            for category in category_totals
        )
        proposed = category_val_counts + group_counts
        val_error = sum(
            abs(proposed[category] - targets[category]) for category in category_totals
        )
        assignments[group_id] = "val" if val_error < current_error else "train"
        if assignments[group_id] == "val":
            category_val_counts.update(group_counts)

    if not assignments or "val" not in assignments.values() or "train" not in assignments.values():
        raise RuntimeError("Sequence-aware split did not produce both train and validation groups")
    return assignments


def mask_to_yolo_boxes(mask: np.ndarray) -> list[dict]:
    binary = (mask > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = mask.shape[:2]
    boxes: list[dict] = []

    for component_id in range(1, component_count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_PIXELS:
            continue
        boxes.append(
            {
                "class_id": FIRE_CLASS_ID,
                "xc": (x + box_width / 2) / width,
                "yc": (y + box_height / 2) / height,
                "w": box_width / width,
                "h": box_height / height,
                "area_pixels": area,
                "area_fraction": area / (width * height),
            }
        )
    return boxes


def validate_pairs(root: Path) -> dict:
    report: dict[str, object] = {"splits": {}, "missing_pair_count": 0, "invalid_rows": []}
    for split in ("train", "val"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        report["missing_pair_count"] = int(report["missing_pair_count"]) + len(missing_labels) + len(missing_images)
        class_counts = Counter()
        box_count = 0
        empty_labels = 0
        boxes_per_image: list[int] = []
        for stem, label_path in labels.items():
            rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            boxes_per_image.append(len(rows))
            if not rows:
                empty_labels += 1
            for row in rows:
                valid = len(row) == 5
                if valid:
                    try:
                        class_id = int(row[0])
                        values = [float(value) for value in row[1:]]
                        valid = class_id in CLASS_NAMES and all(0.0 <= value <= 1.0 for value in values) and values[2] > 0 and values[3] > 0
                    except ValueError:
                        valid = False
                if not valid:
                    report["invalid_rows"].append({"split": split, "label": label_path.name, "row": row})
                else:
                    class_counts[class_id] += 1
                    box_count += 1
        report["splits"][split] = {
            "images": len(images), "labels": len(labels), "empty_labels": empty_labels,
            "boxes": box_count, "class_distribution": dict(sorted(class_counts.items())),
            "missing_labels": len(missing_labels), "missing_images": len(missing_images),
            "boxes_per_image": {"min": min(boxes_per_image, default=0), "max": max(boxes_per_image, default=0), "mean": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0},
        }
    report["invalid_row_count"] = len(report["invalid_rows"])
    return report


def build_multifire(allow_existing: bool = False) -> tuple[list[dict], dict]:
    reusing_existing = MULTIFIRE_OUTPUT.exists()
    if reusing_existing and not allow_existing:
        fail_if_exists(MULTIFIRE_OUTPUT)
    df = pd.read_csv(CSV_PATH)
    selected = df.loc[
        (df["fire_type"] == "fire") & (df["split"] == "train") & (df["label_type"] == "manual")
    ].copy()
    selected = selected.sort_values(["category", "image_name"], kind="stable")
    if len(selected) != 6109:
        raise RuntimeError(f"Unexpected MultiFire selection count: {len(selected)} (expected 6109)")
    test_keys = {
        (str(row.category), str(row.image_name))
        for row in df.loc[df["split"] == "test", ["category", "image_name"]].itertuples(index=False)
    }
    held_out_key_collisions = selected.loc[
        selected.apply(lambda row: (str(row["category"]), str(row["image_name"])) in test_keys, axis=1)
    ].copy()
    # The official metadata is authoritative. Exclude an apparent collision even
    # if a corresponding file is absent in the local test extraction.
    selected = selected.loc[
        ~selected.apply(lambda row: (str(row["category"]), str(row["image_name"])) in test_keys, axis=1)
    ].copy()

    records: list[dict] = []
    for source_index, row in selected.iterrows():
        category = str(row["category"])
        image_name = str(row["image_name"])
        if category not in CATEGORY_FOLDER:
            raise ValueError(f"Unexpected MultiFire category: {category}")
        source_video = video_id(image_name)
        destination_stem = f"mf_{category}_{source_video}_{source_index:05d}_{Path(image_name).stem}"
        records.append({
            "source_index": int(source_index), "original_filename": image_name,
            "source_category": category, "source_video": source_video,
            "source_image": MULTIFIRE_TRAIN_ROOT / CATEGORY_FOLDER[category] / image_name,
            "source_mask": MULTIFIRE_TRAIN_ROOT / CATEGORY_FOLDER[category] / Path(image_name).with_suffix(".tif"),
            "destination_filename": f"{destination_stem}.jpg", "label_type": str(row["label_type"]),
            "official_split": str(row["split"]), "fire_type": str(row["fire_type"]),
        })

    assignments = split_by_video(records)
    for record in records:
        record["dataset_split"] = assignments[record["source_video"]]

    if not reusing_existing:
        MULTIFIRE_OUTPUT.mkdir(parents=True)
    for split in ("train", "val"):
        (MULTIFIRE_OUTPUT / split / "images").mkdir(parents=True, exist_ok=reusing_existing)
        (MULTIFIRE_OUTPUT / split / "labels").mkdir(parents=True, exist_ok=reusing_existing)

    missing_images: list[dict] = []
    missing_masks: list[dict] = []
    unreadable_images: list[dict] = []
    unreadable_masks: list[dict] = []
    empty_masks: list[dict] = []
    prepared: list[dict] = []
    box_areas: list[float] = []

    for record in records:
        image_path = record["source_image"]
        mask_path = record["source_mask"]
        if not image_path.exists():
            missing_images.append({"source_index": record["source_index"], "path": str(image_path)})
            continue
        if not mask_path.exists():
            missing_masks.append({"source_index": record["source_index"], "path": str(mask_path)})
            continue
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            unreadable_images.append({"source_index": record["source_index"], "path": str(image_path)})
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            unreadable_masks.append({"source_index": record["source_index"], "path": str(mask_path)})
            continue
        boxes = mask_to_yolo_boxes(mask)
        if not boxes:
            empty_masks.append({"source_index": record["source_index"], "original_filename": record["original_filename"]})
        split_root = MULTIFIRE_OUTPUT / record["dataset_split"]
        destination_image = split_root / "images" / record["destination_filename"]
        destination_label = split_root / "labels" / f"{Path(record['destination_filename']).stem}.txt"
        expected_label = "".join(
            f"{box['class_id']} {box['xc']:.6f} {box['yc']:.6f} {box['w']:.6f} {box['h']:.6f}\n"
            for box in boxes
        )
        if reusing_existing:
            if not destination_image.exists() or not destination_label.exists():
                raise FileNotFoundError(f"Incomplete prior build: {destination_image} / {destination_label}")
            if destination_label.read_text(encoding="utf-8") != expected_label:
                raise RuntimeError(f"Prior generated label does not match source mask: {destination_label}")
        else:
            shutil.copy2(image_path, destination_image)
            destination_label.write_text(expected_label, encoding="utf-8")
        record = {key: (str(value) if isinstance(value, Path) else value) for key, value in record.items()}
        record["generated_box_count"] = len(boxes)
        record["fire_class_id"] = FIRE_CLASS_ID
        prepared.append(record)
        box_areas.extend(box["area_fraction"] for box in boxes)

    selected_keys = {(record["source_category"], record["original_filename"]) for record in prepared}
    held_out_overlap = sum(key in selected_keys for key in test_keys)
    if held_out_overlap:
        raise RuntimeError(f"MultiFire test leakage detected: {held_out_overlap} selected records also occur in test")

    validation = validate_pairs(MULTIFIRE_OUTPUT)
    train_videos = {record["source_video"] for record in prepared if record["dataset_split"] == "train"}
    val_videos = {record["source_video"] for record in prepared if record["dataset_split"] == "val"}
    video_overlap = sorted(train_videos & val_videos)
    if validation["missing_pair_count"] or validation["invalid_row_count"] or video_overlap:
        raise RuntimeError("MultiFire validation gate failed; output has been retained for inspection")

    report = {
        "selection": {"filter": "fire_type=fire, split=train, label_type=manual", "selected_records_before_test_exclusion": 6109, "excluded_metadata_test_key_collisions": len(held_out_key_collisions), "selected_records_after_test_exclusion": len(records), "urban": sum(r["source_category"] == "urban" for r in records), "rural": sum(r["source_category"] == "rural" for r in records)},
        "class_mapping": CLASS_NAMES, "fire_class_id": FIRE_CLASS_ID, "min_component_pixels": MIN_COMPONENT_PIXELS,
        "prepared_images": len(prepared), "missing_images": missing_images, "missing_masks": missing_masks,
        "unreadable_images": unreadable_images, "unreadable_masks": unreadable_masks, "empty_masks": empty_masks,
        "duplicate_destination_filenames": len(prepared) - len({r["destination_filename"] for r in prepared}),
        "box_area_fraction": {"count": len(box_areas), "min": float(np.min(box_areas)) if box_areas else None, "median": float(np.median(box_areas)) if box_areas else None, "mean": float(np.mean(box_areas)) if box_areas else None, "p90": float(np.percentile(box_areas, 90)) if box_areas else None, "max": float(np.max(box_areas)) if box_areas else None},
        "train_val": {"seed": SEED, "target_val_ratio": VAL_RATIO, "train_videos": sorted(train_videos), "val_videos": sorted(val_videos), "video_overlap": video_overlap, "train_records": sum(r["dataset_split"] == "train" for r in prepared), "val_records": sum(r["dataset_split"] == "val" for r in prepared)},
        "held_out_test_overlap": held_out_overlap, "pair_validation": validation,
    }
    report["excluded_metadata_test_key_collision_samples"] = [
        {"source_index": int(index), "category": str(row["category"]), "original_filename": str(row["image_name"])}
        for index, row in held_out_key_collisions.iterrows()
    ]
    write_json(MULTIFIRE_OUTPUT / "manifest.json", prepared)
    write_json(MULTIFIRE_OUTPUT / "validation_report.json", report)
    write_yaml(MULTIFIRE_OUTPUT / "data.yaml", MULTIFIRE_OUTPUT)
    return prepared, report


def build_combined(multifire_records: list[dict]) -> dict:
    fail_if_exists(COMBINED_OUTPUT)
    COMBINED_OUTPUT.mkdir(parents=True)
    for split in ("train", "val"):
        (COMBINED_OUTPUT / split / "images").mkdir(parents=True)
        (COMBINED_OUTPUT / split / "labels").mkdir(parents=True)

    manifest: list[dict] = []
    for split in ("train", "val"):
        image_dir = DFIRE_SPLIT_ROOT / split / "images"
        label_dir = DFIRE_SPLIT_ROOT / split / "labels"
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"D-Fire image has no label: {image_path}")
            destination_stem = f"dfire_{image_path.stem}"
            shutil.copy2(image_path, COMBINED_OUTPUT / split / "images" / f"{destination_stem}.jpg")
            shutil.copy2(label_path, COMBINED_OUTPUT / split / "labels" / f"{destination_stem}.txt")
            manifest.append({"source": "D-Fire", "source_split": split, "original_filename": image_path.name, "destination_filename": f"{destination_stem}.jpg", "dataset_split": split})

    for record in multifire_records:
        source_root = MULTIFIRE_OUTPUT / record["dataset_split"]
        source_image = source_root / "images" / record["destination_filename"]
        source_label = source_root / "labels" / f"{Path(record['destination_filename']).stem}.txt"
        destination_stem = f"multifire_{Path(record['destination_filename']).stem}"
        split = record["dataset_split"]
        shutil.copy2(source_image, COMBINED_OUTPUT / split / "images" / f"{destination_stem}.jpg")
        shutil.copy2(source_label, COMBINED_OUTPUT / split / "labels" / f"{destination_stem}.txt")
        manifest.append({"source": "MultiFire20K", "source_split": "train", "original_filename": record["original_filename"], "source_category": record["source_category"], "source_video": record["source_video"], "destination_filename": f"{destination_stem}.jpg", "dataset_split": split})

    validation = validate_pairs(COMBINED_OUTPUT)
    if validation["missing_pair_count"] or validation["invalid_row_count"]:
        raise RuntimeError("Combined dataset validation gate failed; output has been retained for inspection")
    report = {"class_mapping": CLASS_NAMES, "sources": {"D-Fire": {"root": str(DFIRE_SPLIT_ROOT), "splits_used": ["train", "val"], "test_used": False}, "MultiFire20K": {"source_filter": "fire_type=fire, split=train, label_type=manual", "test_used": False}}, "pair_validation": validation, "manifest_records": len(manifest)}
    write_json(COMBINED_OUTPUT / "manifest.json", manifest)
    write_json(COMBINED_OUTPUT / "validation_report.json", report)
    write_yaml(COMBINED_OUTPUT / "data.yaml", COMBINED_OUTPUT)
    return report


def main() -> None:
    for required in (CSV_PATH, MULTIFIRE_TRAIN_ROOT, DFIRE_SPLIT_ROOT):
        if not required.exists():
            raise FileNotFoundError(f"Required input does not exist: {required}")
    allowed_args = {"--finalize-existing"}
    if any(argument not in allowed_args for argument in sys.argv[1:]):
        raise ValueError("Only supported argument is --finalize-existing")
    multifire_records, multifire_report = build_multifire("--finalize-existing" in sys.argv)
    combined_report = build_combined(multifire_records)
    print(json.dumps({"multifire": multifire_report, "combined": combined_report}, indent=2))


if __name__ == "__main__":
    main()
