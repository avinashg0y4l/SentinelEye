from pathlib import Path
import json


ROOT = Path(
    "data/experiments/detector_v1/"
    "multifire_urban_test/FU"
)

LABEL_DIR = ROOT / "labels"


def main():
    areas = []

    for label_path in LABEL_DIR.glob("*.txt"):

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            _, xc, yc, w, h = map(
                float,
                parts
            )

            area = w * h
            areas.append(area)

    small = [
        a for a in areas
        if a < 0.01
    ]

    medium = [
        a for a in areas
        if 0.01 <= a < 0.10
    ]

    large = [
        a for a in areas
        if a >= 0.10
    ]

    total = len(areas)

    report = {
        "total_boxes": total,

        "small": {
            "count": len(small),
            "percent": (
                100 * len(small) / total
                if total else 0
            ),
        },

        "medium": {
            "count": len(medium),
            "percent": (
                100 * len(medium) / total
                if total else 0
            ),
        },

        "large": {
            "count": len(large),
            "percent": (
                100 * len(large) / total
                if total else 0
            ),
        },
    }

    print("=" * 60)
    print("MultiFire20K - Fire Box Size Distribution")
    print("=" * 60)

    print(json.dumps(
        report,
        indent=4
    ))

    output = (
        ROOT.parent
        / "box_size_distribution.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()