from pathlib import Path
import shutil
import json


SOURCE = Path(
    "data/experiments/detector_v2/"
    "multifire_manual_sequence_v2_safe"
)

OUTPUT = Path(
    "data/experiments/detector_v2/"
    "multifire_manual_sequence_v2_clean"
)


def copy_split(split):
    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    out_images = OUTPUT / split / "images"
    out_labels = OUTPUT / split / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    total = 0
    removed = 0
    boxes = 0

    for image_path in sorted(src_images.glob("*")):

        if not image_path.is_file():
            continue

        label_path = src_labels / f"{image_path.stem}.txt"

        total += 1

        # Remove empty-label training samples.
        if split == "train":
            if not label_path.exists():
                removed += 1
                continue

            content = label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if not content:
                removed += 1
                continue

        if not label_path.exists():
            raise RuntimeError(
                f"Missing label for {image_path.name}"
            )

        shutil.copy2(
            image_path,
            out_images / image_path.name
        )

        shutil.copy2(
            label_path,
            out_labels / label_path.name
        )

        lines = [
            x.strip()
            for x in label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()
            if x.strip()
        ]

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                raise RuntimeError(
                    f"Invalid YOLO label: {label_path}"
                )

            cls = int(parts[0])

            if cls not in (0, 1):
                raise RuntimeError(
                    f"Invalid class {cls}: {label_path}"
                )

            boxes += 1

    return {
        "source_images": total,
        "removed_empty": removed,
        "output_images": total - removed,
        "boxes": boxes,
    }


def main():
    print("=" * 70)
    print("SentinelEye - Clean MultiFire V2 Dataset")
    print("=" * 70)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source not found: {SOURCE}"
        )

    train = copy_split("train")
    val = copy_split("val")

    report = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "train": train,
        "val": val,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    (OUTPUT / "cleaning_report.json").write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
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

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"Train source : {train['source_images']}"
    )
    print(
        f"Train removed: {train['removed_empty']}"
    )
    print(
        f"Train output  : {train['output_images']}"
    )
    print(
        f"Train boxes   : {train['boxes']}"
    )

    print()

    print(
        f"Val source   : {val['source_images']}"
    )
    print(
        f"Val output   : {val['output_images']}"
    )
    print(
        f"Val boxes    : {val['boxes']}"
    )

    print("\nOutput:")
    print(OUTPUT)


if __name__ == "__main__":
    main()