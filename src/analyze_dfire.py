from pathlib import Path
from collections import Counter

ROOT = Path("data/processed/DFIRE")

CLASS_NAMES = {
    0: "Fire",
    1: "Smoke",
}


def analyze_split(split_name: str):
    image_dir = ROOT / split_name / "images"
    label_dir = ROOT / split_name / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file()
    ]

    labels = [
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    class_counts = Counter()
    total_boxes = 0
    empty_labels = 0

    for label_file in labels:
        text = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            empty_labels += 1
            continue

        for line in text.splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            class_counts[class_id] += 1
            total_boxes += 1

    print("\n" + "=" * 55)
    print(split_name.upper())
    print("=" * 55)

    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Empty labels:", empty_labels)
    print("Total bounding boxes:", total_boxes)

    print("\nBounding boxes by class:")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_name}: "
            f"{class_counts[class_id]}"
        )


print("D-Fire Dataset Analysis")

analyze_split("train")
analyze_split("test")