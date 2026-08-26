from pathlib import Path
import json


ROOT = Path(
    "data/experiments/detector_v2/"
    "combined_v2_clean"
)


def validate_split(split):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = {
        p.stem
        for p in image_dir.glob("*")
        if p.is_file()
    }

    labels = {
        p.stem
        for p in label_dir.glob("*.txt")
    }

    missing_labels = sorted(images - labels)
    orphan_labels = sorted(labels - images)

    invalid_rows = 0
    class_counts = {0: 0, 1: 0}
    empty_labels = 0

    for label_path in label_dir.glob("*.txt"):

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid_rows += 1
                continue

            try:
                cls = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                invalid_rows += 1
                continue

            if cls not in (0, 1):
                invalid_rows += 1
                continue

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_rows += 1
                continue

            class_counts[cls] += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "invalid_rows": invalid_rows,
        "empty_labels": empty_labels,
        "class_counts": {
            "0_Smoke": class_counts[0],
            "1_Fire": class_counts[1],
        },
        "missing_label_examples": missing_labels[:10],
        "orphan_label_examples": orphan_labels[:10],
    }


def main():

    print("=" * 70)
    print("SentinelEye - Combined V2 Integrity Check")
    print("=" * 70)

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {ROOT}"
        )

    train = validate_split("train")
    val = validate_split("val")

    train_names = {
        p.stem
        for p in (ROOT / "train" / "images").glob("*")
        if p.is_file()
    }

    val_names = {
        p.stem
        for p in (ROOT / "val" / "images").glob("*")
        if p.is_file()
    }

    overlap = train_names & val_names

    report = {
        "train": train,
        "val": val,
        "train_val_filename_overlap": len(overlap),
        "overlap_examples": sorted(overlap)[:20],
    }

    print("\nTRAIN")
    print(json.dumps(train, indent=4))

    print("\nVALIDATION")
    print(json.dumps(val, indent=4))

    print("\nTRAIN/VAL OVERLAP:")
    print(len(overlap))

    output = ROOT / "integrity_report.json"

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(output)

    if (
        train["missing_labels"] == 0
        and train["orphan_labels"] == 0
        and train["invalid_rows"] == 0
        and val["missing_labels"] == 0
        and val["orphan_labels"] == 0
        and val["invalid_rows"] == 0
        and len(overlap) == 0
    ):
        print("\nSTATUS: PASS")
    else:
        print("\nSTATUS: FAIL")


if __name__ == "__main__":
    main()