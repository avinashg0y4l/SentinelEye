from pathlib import Path
import json
import numpy as np


ROOT = Path(
    "data/experiments/detector_v1/"
    "multifire_urban_test/FU"
)

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"


def main():

    image_files = list(
        IMAGE_DIR.glob("*.jpg")
    )

    box_counts = []
    box_areas = []
    aspect_ratios = []
    widths = []
    heights = []

    empty = 0

    for image_path in image_files:

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            empty += 1
            continue

        lines = [
            x.strip()
            for x in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

        if not lines:
            empty += 1
            continue

        box_counts.append(len(lines))

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            _, xc, yc, w, h = map(
                float,
                parts
            )

            box_areas.append(w * h)

            if h > 0:
                aspect_ratios.append(w / h)

            widths.append(w)
            heights.append(h)

    def percentile(values, p):
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    report = {
        "images": len(image_files),
        "empty_labels": empty,
        "total_boxes": int(sum(box_counts)),
        "images_with_boxes": len(box_counts),

        "boxes_per_image": {
            "mean": float(np.mean(box_counts)) if box_counts else 0,
            "median": percentile(box_counts, 50),
            "p90": percentile(box_counts, 90),
            "max": int(max(box_counts)) if box_counts else 0,
        },

        "box_area_fraction": {
            "mean": float(np.mean(box_areas)) if box_areas else 0,
            "median": percentile(box_areas, 50),
            "p10": percentile(box_areas, 10),
            "p25": percentile(box_areas, 25),
            "p75": percentile(box_areas, 75),
            "p90": percentile(box_areas, 90),
        },

        "box_width_fraction": {
            "mean": float(np.mean(widths)) if widths else 0,
            "median": percentile(widths, 50),
        },

        "box_height_fraction": {
            "mean": float(np.mean(heights)) if heights else 0,
            "median": percentile(heights, 50),
        },

        "aspect_ratio": {
            "mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0,
            "median": percentile(aspect_ratios, 50),
        },
    }

    print("=" * 65)
    print("MultiFire20K - Box Statistics")
    print("=" * 65)

    print(json.dumps(report, indent=4))

    output = ROOT.parent / "box_statistics.json"

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