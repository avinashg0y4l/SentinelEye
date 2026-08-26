"""Derive cleaned detector-v2 datasets by excluding reviewed empty masks.

This is a non-destructive derivation step. It refuses to overwrite outputs and
does not read or write any MultiFire test data or model artifacts.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


SOURCE_MF = Path("data/experiments/detector_v2/multifire_manual_sequence_v2_safe")
SOURCE_DFIRE = Path("data/experiments/detector_v1/DFIRE_clean_split")
OUTPUT_MF = Path("data/experiments/detector_v2/multifire_manual_sequence_v2_safe_clean")
OUTPUT_COMBINED = Path("data/experiments/detector_v2/combined_v2_safe_clean")
CLASS_NAMES = {0: "Smoke", 1: "Fire"}


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_yaml(path: Path, root: Path) -> None:
    path.write_text(
        f"path: {root.resolve().as_posix()}\n\ntrain: train/images\nval: val/images\n\nnames:\n  0: Smoke\n  1: Fire\n",
        encoding="utf-8",
    )


def read_rows(label_path: Path) -> list[list[str]]:
    return [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(root: Path) -> dict:
    report = {"splits": {}, "missing_pair_count": 0, "invalid_row_count": 0}
    for split in ("train", "val"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = {p.stem: p for p in image_dir.glob("*.jpg")}
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        report["missing_pair_count"] += len(missing_labels) + len(missing_images)
        classes = Counter()
        boxes_per_image = []
        empty = 0
        invalid = []
        for stem, label_path in labels.items():
            rows = read_rows(label_path)
            boxes_per_image.append(len(rows))
            if not rows:
                empty += 1
            for row in rows:
                valid = len(row) == 5
                if valid:
                    try:
                        cls = int(row[0])
                        values = [float(v) for v in row[1:]]
                        valid = cls in CLASS_NAMES and all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0
                    except ValueError:
                        valid = False
                if not valid:
                    invalid.append({"label": label_path.name, "row": row})
                else:
                    classes[int(row[0])] += 1
        report["invalid_row_count"] += len(invalid)
        report["splits"][split] = {
            "images": len(images), "labels": len(labels), "empty_labels": empty,
            "boxes": sum(classes.values()), "class_distribution": dict(sorted(classes.items())),
            "missing_labels": len(missing_labels), "missing_images": len(missing_images),
            "invalid_rows": invalid,
            "boxes_per_image": {"min": min(boxes_per_image, default=0), "max": max(boxes_per_image, default=0), "mean": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0},
        }
    return report


def derive_multifire() -> tuple[list[dict], dict]:
    refuse_existing(OUTPUT_MF)
    source_manifest = json.loads((SOURCE_MF / "manifest.json").read_text(encoding="utf-8"))
    removed = [record for record in source_manifest if record.get("generated_box_count") == 0]
    included = [record for record in source_manifest if record.get("generated_box_count") != 0]
    if len(removed) != 36:
        raise RuntimeError(f"Expected exactly 36 empty-mask records, found {len(removed)}")
    if len(included) != 5872:
        raise RuntimeError(f"Expected 5,872 retained records, found {len(included)}")

    OUTPUT_MF.mkdir(parents=True)
    for split in ("train", "val"):
        (OUTPUT_MF / split / "images").mkdir(parents=True)
        (OUTPUT_MF / split / "labels").mkdir(parents=True)

    for record in included:
        split = record["dataset_split"]
        stem = Path(record["destination_filename"]).stem
        source_image = SOURCE_MF / split / "images" / record["destination_filename"]
        source_label = SOURCE_MF / split / "labels" / f"{stem}.txt"
        if not source_image.exists() or not source_label.exists():
            raise FileNotFoundError(f"Missing source pair for {record['destination_filename']}")
        shutil.copy2(source_image, OUTPUT_MF / split / "images" / record["destination_filename"])
        shutil.copy2(source_label, OUTPUT_MF / split / "labels" / f"{stem}.txt")

    validation = validate_dataset(OUTPUT_MF)
    train_videos = {r["source_video"] for r in included if r["dataset_split"] == "train"}
    val_videos = {r["source_video"] for r in included if r["dataset_split"] == "val"}
    video_overlap = sorted(train_videos & val_videos)
    duplicate_names = len(included) - len({r["destination_filename"] for r in included})
    report = {
        "source": str(SOURCE_MF), "class_mapping": CLASS_NAMES,
        "source_records": len(source_manifest), "removed_empty_mask_records": len(removed),
        "retained_records": len(included), "removed_records": removed,
        "duplicate_destination_filenames": duplicate_names,
        "train_validation_source_video_overlap": video_overlap,
        "validation": validation,
        "test_data_used": False,
    }
    write_json(OUTPUT_MF / "manifest.json", included)
    write_json(OUTPUT_MF / "validation_report.json", report)
    write_yaml(OUTPUT_MF / "data.yaml", OUTPUT_MF)
    return included, report


def derive_combined(multifire_records: list[dict], mf_report: dict) -> dict:
    refuse_existing(OUTPUT_COMBINED)
    OUTPUT_COMBINED.mkdir(parents=True)
    for split in ("train", "val"):
        (OUTPUT_COMBINED / split / "images").mkdir(parents=True)
        (OUTPUT_COMBINED / split / "labels").mkdir(parents=True)

    combined_manifest = []
    for split in ("train", "val"):
        for image_path in sorted((SOURCE_DFIRE / split / "images").glob("*.jpg")):
            label_path = SOURCE_DFIRE / split / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing D-Fire pair: {image_path.name}")
            stem = f"dfire_{image_path.stem}"
            shutil.copy2(image_path, OUTPUT_COMBINED / split / "images" / f"{stem}.jpg")
            shutil.copy2(label_path, OUTPUT_COMBINED / split / "labels" / f"{stem}.txt")
            combined_manifest.append({"source": "D-Fire", "original_filename": image_path.name, "destination_filename": f"{stem}.jpg", "dataset_split": split})

    for record in multifire_records:
        split = record["dataset_split"]
        stem = Path(record["destination_filename"]).stem
        source_image = OUTPUT_MF / split / "images" / record["destination_filename"]
        source_label = OUTPUT_MF / split / "labels" / f"{stem}.txt"
        output_stem = f"multifire_{stem}"
        shutil.copy2(source_image, OUTPUT_COMBINED / split / "images" / f"{output_stem}.jpg")
        shutil.copy2(source_label, OUTPUT_COMBINED / split / "labels" / f"{output_stem}.txt")
        combined_manifest.append({"source": "MultiFire20K", "original_filename": record["original_filename"], "destination_filename": f"{output_stem}.jpg", "source_category": record["source_category"], "source_video": record["source_video"], "dataset_split": split, "fire_only_supervision": True})

    validation = validate_dataset(OUTPUT_COMBINED)
    train_names = {r["destination_filename"] for r in combined_manifest if r["dataset_split"] == "train"}
    val_names = {r["destination_filename"] for r in combined_manifest if r["dataset_split"] == "val"}
    filename_overlap = sorted(train_names & val_names)
    train_videos = {r["source_video"] for r in multifire_records if r["dataset_split"] == "train"}
    val_videos = {r["source_video"] for r in multifire_records if r["dataset_split"] == "val"}
    video_overlap = sorted(train_videos & val_videos)
    category_counts = Counter(r["source_category"] for r in multifire_records)
    report = {
        "sources": {"D-Fire": {"root": str(SOURCE_DFIRE), "test_used": False}, "MultiFire20K": {"root": str(OUTPUT_MF), "test_used": False, "fire_only": True}},
        "class_mapping": CLASS_NAMES, "validation": validation,
        "train_validation_filename_overlap": filename_overlap,
        "train_validation_multifire_source_video_overlap": video_overlap,
        "multifire_category_distribution": dict(sorted(category_counts.items())),
        "manifest_records": len(combined_manifest), "removed_empty_mask_records": mf_report["removed_empty_mask_records"],
        "test_data_used": False,
    }
    write_json(OUTPUT_COMBINED / "manifest.json", combined_manifest)
    write_json(OUTPUT_COMBINED / "validation_report.json", report)
    write_yaml(OUTPUT_COMBINED / "data.yaml", OUTPUT_COMBINED)
    return report


def main() -> None:
    for required in (SOURCE_MF / "manifest.json", SOURCE_MF / "data.yaml", SOURCE_DFIRE):
        if not required.exists():
            raise FileNotFoundError(required)
    records, mf_report = derive_multifire()
    combined_report = derive_combined(records, mf_report)
    print(json.dumps({"multifire": mf_report, "combined": combined_report}, indent=2))


if __name__ == "__main__":
    main()
