from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import shutil


CSV_PATH = Path(
    "data/raw/Multifire/data_structure.csv"
)

SOURCE_ROOT = Path(
    "data/experiments/detector_v2/"
    "multifire_manual_train/Train"
)

OUTPUT_ROOT = Path(
    "data/experiments/detector_v2/"
    "multifire_manual_train_yolo"
)

IMAGE_OUT = OUTPUT_ROOT / "images"
LABEL_OUT = OUTPUT_ROOT / "labels"

MIN_COMPONENT_PIXELS = 10

# SentinelEye convention
# 0 = Smoke
# 1 = Fire
FIRE_CLASS_ID = 1


def mask_to_boxes(mask):
    """
    Convert a binary fire mask into Fire bounding boxes.

    Each connected component becomes one Fire box.
    """

    binary = (
        mask > 0
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    height, width = mask.shape

    boxes = []

    for component_id in range(
        1,
        num_labels,
    ):

        x = stats[
            component_id,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            component_id,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            component_id,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            component_id,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_COMPONENT_PIXELS:
            continue

        x_center = (
            x + w / 2
        ) / width

        y_center = (
            y + h / 2
        ) / height

        box_width = (
            w / width
        )

        box_height = (
            h / height
        )

        boxes.append({
            "class_id": FIRE_CLASS_ID,
            "xc": x_center,
            "yc": y_center,
            "w": box_width,
            "h": box_height,
            "area_pixels": int(area),
            "area_fraction": (
                float(area)
                / float(width * height)
            ),
        })

    return boxes


def main():

    print("=" * 70)
    print(
        "SentinelEye - MultiFire20K "
        "Manual Training Preparation"
    )
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}"
        )

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_ROOT}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    # EXACT selection requested:
    # Fire + Train + Manual
    selected = df[
        (df["fire_type"] == "fire")
        & (df["split"] == "train")
        & (df["label_type"] == "manual")
    ].copy()

    print(
        f"Selected manual Fire training images: "
        f"{len(selected)}"
    )

    print("\nBy category:")
    print(
        selected["category"]
        .value_counts()
        .to_string()
    )

    IMAGE_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LABEL_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_selected = len(selected)

    copied_images = 0
    created_labels = 0
    total_boxes = 0

    urban_images = 0
    rural_images = 0

    missing_images = []
    missing_masks = []

    empty_masks = []
    unreadable_masks = []

    box_area_fraction = []

    manifest = []

    for row_number, row in selected.iterrows():

        image_name = str(
            row["image_name"]
        )

        category = str(
            row["category"]
        )

        image_path = (
            SOURCE_ROOT
            / (
                "FU"
                if category == "urban"
                else "FR"
            )
            / image_name
        )

        mask_path = (
            image_path.with_suffix(".tif")
        )

        # -------------------------------------------------
        # Check image
        # -------------------------------------------------

        if not image_path.exists():

            missing_images.append(
                image_name
            )

            continue

        # -------------------------------------------------
        # Check mask
        # -------------------------------------------------

        if not mask_path.exists():

            missing_masks.append(
                image_name
            )

            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:

            unreadable_masks.append(
                image_name
            )

            continue

        boxes = mask_to_boxes(
            mask
        )

        if not boxes:

            empty_masks.append(
                image_name
            )

        # -------------------------------------------------
        # Copy image
        # -------------------------------------------------

        destination_image = (
            IMAGE_OUT
            / image_name
        )

        shutil.copy2(
            image_path,
            destination_image,
        )

        copied_images += 1

        # -------------------------------------------------
        # Create YOLO label
        # -------------------------------------------------

        label_path = (
            LABEL_OUT
            / f"{Path(image_name).stem}.txt"
        )

        with label_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            for box in boxes:

                f.write(
                    f"{box['class_id']} "
                    f"{box['xc']:.6f} "
                    f"{box['yc']:.6f} "
                    f"{box['w']:.6f} "
                    f"{box['h']:.6f}\n"
                )

                total_boxes += 1

                box_area_fraction.append(
                    box["area_fraction"]
                )

        created_labels += 1

        if category == "urban":
            urban_images += 1
        else:
            rural_images += 1

        manifest.append({
            "image_name": image_name,
            "category": category,
            "label_type": str(
                row["label_type"]
            ),
            "fire_type": str(
                row["fire_type"]
            ),
            "split": str(
                row["split"]
            ),
            "boxes": len(boxes),
        })

        if (
            copied_images > 0
            and copied_images % 500 == 0
        ):
            print(
                f"Processed: "
                f"{copied_images}/{total_selected}"
            )

    # -----------------------------------------------------
    # Dataset YAML
    # -----------------------------------------------------

    yaml_path = (
        OUTPUT_ROOT
        / "data.yaml"
    )

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}

train: images
val: images

names:
  0: Smoke
  1: Fire
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    box_area_stats = {}

    if box_area_fraction:

        arr = np.array(
            box_area_fraction,
            dtype=float,
        )

        box_area_stats = {
            "mean": float(
                np.mean(arr)
            ),
            "median": float(
                np.median(arr)
            ),
            "p10": float(
                np.percentile(arr, 10)
            ),
            "p25": float(
                np.percentile(arr, 25)
            ),
            "p75": float(
                np.percentile(arr, 75)
            ),
            "p90": float(
                np.percentile(arr, 90)
            ),
            "max": float(
                np.max(arr)
            ),
        }

    report = {
        "source": str(
            SOURCE_ROOT
        ),

        "csv": str(
            CSV_PATH
        ),

        "selected_images": total_selected,

        "copied_images": copied_images,

        "created_labels": created_labels,

        "total_fire_boxes": total_boxes,

        "urban_images": urban_images,

        "rural_images": rural_images,

        "missing_images": len(
            missing_images
        ),

        "missing_masks": len(
            missing_masks
        ),

        "empty_masks": len(
            empty_masks
        ),

        "unreadable_masks": len(
            unreadable_masks
        ),

        "min_component_pixels":
            MIN_COMPONENT_PIXELS,

        "box_area_fraction":
            box_area_stats,

        "missing_image_names":
            missing_images[:50],

        "missing_mask_names":
            missing_masks[:50],

        "empty_mask_names":
            empty_masks[:50],

        "unreadable_mask_names":
            unreadable_masks[:50],
    }

    report_path = (
        OUTPUT_ROOT
        / "preparation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    manifest_path = (
        OUTPUT_ROOT
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(
        "Selected:",
        total_selected
    )

    print(
        "Images copied:",
        copied_images
    )

    print(
        "Labels created:",
        created_labels
    )

    print(
        "Fire boxes:",
        total_boxes
    )

    print(
        "Urban:",
        urban_images
    )

    print(
        "Rural:",
        rural_images
    )

    print(
        "Missing images:",
        len(missing_images)
    )

    print(
        "Missing masks:",
        len(missing_masks)
    )

    print(
        "Empty masks:",
        len(empty_masks)
    )

    print(
        "Unreadable masks:",
        len(unreadable_masks)
    )

    print("\nDataset YAML:")
    print(yaml_path)

    print("\nReport:")
    print(report_path)

    print("\nManifest:")
    print(manifest_path)


if __name__ == "__main__":
    main()