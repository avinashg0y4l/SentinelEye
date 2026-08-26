from pathlib import Path

ROOT = Path("data/processed/DFIRE")

SPLITS = ["train", "test"]


def check_split(split):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    invalid_boxes = []
    total_objects = 0
    empty_labels = 0

    for label_file in labels:

        text = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            empty_labels += 1
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                invalid_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        "Expected 5 values"
                    )
                )
                continue

            try:
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]
            except ValueError:
                invalid_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        "Non-numeric value"
                    )
                )
                continue

            x, y, w, h = values

            if class_id not in [0, 1]:
                invalid_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        f"Invalid class {class_id}"
                    )
                )

            if not all(
                0.0 <= value <= 1.0
                for value in [x, y, w, h]
            ):
                invalid_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        "Coordinate outside 0-1"
                    )
                )

            if w <= 0 or h <= 0:
                invalid_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        "Width/height <= 0"
                    )
                )

            total_objects += 1

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Empty labels:", empty_labels)
    print("Objects:", total_objects)
    print("Missing labels:", len(missing_labels))
    print("Orphan labels:", len(orphan_labels))
    print("Invalid annotations:", len(invalid_boxes))

    if missing_labels:
        print("\nExample missing labels:")
        print(list(sorted(missing_labels))[:5])

    if orphan_labels:
        print("\nExample orphan labels:")
        print(list(sorted(orphan_labels))[:5])

    if invalid_boxes:
        print("\nExample invalid annotations:")
        for item in invalid_boxes[:5]:
            print(item)


print("SentinelEye - D-Fire YOLO Preflight Check")

for split in SPLITS:
    check_split(split)