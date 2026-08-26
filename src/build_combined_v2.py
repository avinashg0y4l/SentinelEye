from pathlib import Path
import shutil
import json


DFIRE = Path(
    "data/experiments/detector_v1/"
    "DFIRE_clean_split"
)

MULTIFIRE = Path(
    "data/experiments/detector_v2/"
    "multifire_manual_sequence_v2_clean"
)

OUTPUT = Path(
    "data/experiments/detector_v2/"
    "combined_v2_clean"
)


def copy_dataset(source_root, split, output_root):
    src_images = source_root / split / "images"
    src_labels = source_root / split / "labels"

    dst_images = output_root / split / "images"
    dst_labels = output_root / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    boxes = {
        0: 0,
        1: 0,
    }

    for image_path in sorted(src_images.glob("*")):

        if not image_path.is_file():
            continue

        label_path = (
            src_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise RuntimeError(
                f"Missing label: {label_path}"
            )

        destination_image = (
            dst_images / image_path.name
        )

        destination_label = (
            dst_labels / label_path.name
        )

        if destination_image.exists():
            raise RuntimeError(
                f"Filename collision: "
                f"{image_path.name}"
            )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        for line in label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():

            parts = line.split()

            if not parts:
                continue

            if len(parts) != 5:
                raise RuntimeError(
                    f"Invalid YOLO row: {label_path}"
                )

            cls = int(parts[0])

            if cls not in (0, 1):
                raise RuntimeError(
                    f"Invalid class {cls}: {label_path}"
                )

            boxes[cls] += 1

        count += 1

    return count, boxes


def main():

    print("=" * 70)
    print("SentinelEye - Build Combined V2 Dataset")
    print("=" * 70)

    if OUTPUT.exists():
        raise RuntimeError(
            f"Output already exists: {OUTPUT}\n"
            "Delete it manually only if you intend "
            "to rebuild it."
        )

    total = {
        "train": {
            "images": 0,
            "smoke": 0,
            "fire": 0,
        },
        "val": {
            "images": 0,
            "smoke": 0,
            "fire": 0,
        },
    }

    manifest = []

    for split in ("train", "val"):

        print(f"\nProcessing {split.upper()}")

        for name, root in (
            ("DFIRE", DFIRE),
            ("MULTIFIRE", MULTIFIRE),
        ):

            count, boxes = copy_dataset(
                root,
                split,
                OUTPUT
            )

            total[split]["images"] += count
            total[split]["smoke"] += boxes[0]
            total[split]["fire"] += boxes[1]

            manifest.append({
                "dataset": name,
                "split": split,
                "images": count,
                "smoke_boxes": boxes[0],
                "fire_boxes": boxes[1],
            })

            print(
                f"{name}: "
                f"{count} images | "
                f"Smoke={boxes[0]} | "
                f"Fire={boxes[1]}"
            )

    (OUTPUT / "data.yaml").write_text(
        f"""path: {OUTPUT.as_posix()}

train: train/images
val: val/images

names:
  0: Smoke
  1: Fire
""",
        encoding="utf-8"
    )

    report = {
        "output": str(OUTPUT),
        "splits": total,
        "sources": manifest,
    }

    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("COMBINED DATASET READY")
    print("=" * 70)

    for split in ("train", "val"):
        s = total[split]

        print(
            f"{split.upper()}: "
            f"{s['images']} images | "
            f"Smoke={s['smoke']} | "
            f"Fire={s['fire']}"
        )

    print("\nData YAML:")
    print(OUTPUT / "data.yaml")

    print("\nValidation report:")
    print(
        OUTPUT
        / "validation_report.json"
    )


if __name__ == "__main__":
    main()