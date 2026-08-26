from pathlib import Path
import shutil
import json


SOURCE = Path("data/processed/DFIRE/train")
OUTPUT = Path("data/experiments/detector_v1/DFIRE_clean/train")
REPORT_PATH = Path(
    "data/experiments/detector_v1/dfire_sanitization_report.json"
)

CLASS_IDS = {0, 1}


def validate_row(parts):
    """
    Validate a YOLO annotation row.

    Format:
    class_id x_center y_center width height
    """

    if len(parts) != 5:
        return False, "wrong_field_count"

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        return False, "non_numeric"

    if class_id not in CLASS_IDS:
        return False, "invalid_class"

    values = [x_center, y_center, width, height]

    if not all(0.0 <= value <= 1.0 for value in values):
        return False, "coordinate_out_of_range"

    if width <= 0 or height <= 0:
        return False, "zero_or_negative_box"

    return True, None


def main():
    output_images = OUTPUT / "images"
    output_labels = OUTPUT / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    report = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "total_images": 0,
        "total_labels": 0,
        "empty_labels": 0,
        "valid_rows": 0,
        "removed_rows": 0,
        "removed_by_reason": {},
        "affected_files": [],
    }

    image_dir = SOURCE / "images"
    label_dir = SOURCE / "labels"

    image_files = sorted(image_dir.glob("*.jpg"))

    print("Starting D-Fire sanitization...\n")
    print("Images found:", len(image_files))

    for image_path in image_files:

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print("WARNING - missing label:", image_path.name)
            continue

        report["total_images"] += 1
        report["total_labels"] += 1

        # Copy image unchanged
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        text = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        # Empty annotation = valid negative image
        if not text:
            report["empty_labels"] += 1

            shutil.copy2(
                label_path,
                output_labels / label_path.name
            )

            continue

        valid_lines = []
        removed_lines = []

        for line_number, line in enumerate(
            text.splitlines(),
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            valid, reason = validate_row(parts)

            if valid:
                valid_lines.append(line)
                report["valid_rows"] += 1

            else:
                removed_lines.append({
                    "line_number": line_number,
                    "line": line,
                    "reason": reason,
                })

                report["removed_rows"] += 1

                report["removed_by_reason"][reason] = (
                    report["removed_by_reason"].get(reason, 0)
                    + 1
                )

        if removed_lines:
            report["affected_files"].append({
                "file": label_path.name,
                "removed": removed_lines,
            })

        # Write cleaned annotation
        output_label = output_labels / label_path.name

        if valid_lines:
            output_label.write_text(
                "\n".join(valid_lines) + "\n",
                encoding="utf-8"
            )
        else:
            # Image remains a valid negative if all rows were invalid
            output_label.write_text(
                "",
                encoding="utf-8"
            )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("SANITIZATION COMPLETE")
    print("=" * 60)

    print("Images copied:", report["total_images"])
    print("Labels copied:", report["total_labels"])
    print("Empty labels:", report["empty_labels"])
    print("Valid annotation rows:", report["valid_rows"])
    print("Removed annotation rows:", report["removed_rows"])

    print("\nRemoved by reason:")

    for reason, count in report["removed_by_reason"].items():
        print(f"  {reason}: {count}")

    print("\nClean dataset:")
    print(OUTPUT)

    print("\nReport:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()