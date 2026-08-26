from pathlib import Path
import shutil
import json

import cv2
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "data/raw/Multifire/data_structure.csv"
)

SOURCE_ROOT = Path(
    "data/processed/Multifire/Test/Test/FU"
)

OUTPUT_ROOT = Path(
    "data/experiments/detector_v1/"
    "multifire_urban_test/FU"
)

IMAGE_OUT = OUTPUT_ROOT / "images"
LABEL_OUT = OUTPUT_ROOT / "labels"

# D-Fire / SentinelEye class convention:
# 0 = Smoke
# 1 = Fire
FIRE_CLASS_ID = 1

# Ignore extremely tiny mask noise.
MIN_COMPONENT_PIXELS = 10


def mask_to_yolo_boxes(mask):
    """
    Convert binary fire mask into one YOLO box
    for each sufficiently large connected component.
    """

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    height, width = mask.shape

    boxes = []

    for component_id in range(1, num_labels):

        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]

        w = stats[component_id, cv2.CC_STAT_WIDTH]
        h = stats[component_id, cv2.CC_STAT_HEIGHT]

        area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_COMPONENT_PIXELS:
            continue

        # Convert mask coordinates into YOLO format.
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height

        box_width = w / width
        box_height = h / height

        boxes.append(
            (
                FIRE_CLASS_ID,
                x_center,
                y_center,
                box_width,
                box_height,
            )
        )

    return boxes


def main():

    print("=" * 70)
    print("SentinelEye - MultiFire20K FU Test Preparation")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}"
        )

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_ROOT}"
        )

    df = pd.read_csv(CSV_PATH)

    # Official test + urban + fire subset.
    selected = df[
        (df["fire_type"] == "fire")
        & (df["category"] == "urban")
        & (df["split"] == "test")
    ].copy()

    print(
        "Selected FU test images:",
        len(selected)
    )

    manual_count = (
        selected["label_type"]
        .value_counts()
        .to_dict()
    )

    print(
        "Label types:",
        manual_count
    )

    IMAGE_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LABEL_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_images = 0
    total_labels = 0
    empty_masks = 0
    missing_images = 0
    missing_masks = 0
    total_boxes = 0

    component_counts = []

    for _, row in selected.iterrows():

        image_name = str(
            row["image_name"]
        )

        image_path = (
            SOURCE_ROOT / image_name
        )

        mask_path = (
            SOURCE_ROOT
            / Path(image_name).with_suffix(".tif")
        )

        if not image_path.exists():

            missing_images += 1

            print(
                "Missing image:",
                image_path
            )

            continue

        if not mask_path.exists():

            missing_masks += 1

            print(
                "Missing mask:",
                mask_path
            )

            continue

        # Read mask.
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:

            print(
                "Could not read mask:",
                mask_path
            )

            missing_masks += 1

            continue

        boxes = mask_to_yolo_boxes(mask)

        # Copy original image.
        shutil.copy2(
            image_path,
            IMAGE_OUT / image_name,
        )

        label_path = (
            LABEL_OUT
            / f"{Path(image_name).stem}.txt"
        )

        with open(
            label_path,
            "w",
            encoding="utf-8",
        ) as f:

            for (
                class_id,
                xc,
                yc,
                bw,
                bh,
            ) in boxes:

                f.write(
                    f"{class_id} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}\n"
                )

        total_images += 1
        total_boxes += len(boxes)

        if not boxes:
            empty_masks += 1
        else:
            component_counts.append(
                len(boxes)
            )

        total_labels += 1

    # Dataset YAML.
    yaml_path = (
        OUTPUT_ROOT.parent
        / "data.yaml"
    )

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}

train: images
val: images
test: images

names:
  0: Smoke
  1: Fire
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    report = {
        "selected_csv_rows": len(selected),
        "processed_images": total_images,
        "labels_created": total_labels,
        "total_fire_boxes": total_boxes,
        "empty_masks": empty_masks,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "manual_label_type_counts": manual_count,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
    }

    report_path = (
        OUTPUT_ROOT.parent
        / "preparation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(
        "CSV selected:",
        len(selected)
    )

    print(
        "Images copied:",
        total_images
    )

    print(
        "Labels created:",
        total_labels
    )

    print(
        "Fire boxes:",
        total_boxes
    )

    print(
        "Empty masks:",
        empty_masks
    )

    print(
        "Missing images:",
        missing_images
    )

    print(
        "Missing masks:",
        missing_masks
    )

    print("\nDataset YAML:")
    print(yaml_path)

    print("\nReport:")
    print(report_path)


if __name__ == "__main__":
    main()