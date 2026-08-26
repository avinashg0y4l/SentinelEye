from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


CSV_PATH = Path(
    "data/raw/Multifire/data_structure.csv"
)

SOURCE_ROOT = Path(
    "data/processed/Multifire/Test/Test/FU"
)

OUTPUT = Path(
    "data/experiments/detector_v1/"
    "multifire_urban_test/mask_analysis.json"
)

MIN_COMPONENT_PIXELS = 10


def main():

    import pandas as pd

    df = pd.read_csv(CSV_PATH)

    selected = df[
        (df["fire_type"] == "fire")
        & (df["category"] == "urban")
        & (df["split"] == "test")
    ]

    mask_counts = []
    mask_area = []
    component_area = []
    component_counts = []

    empty_masks = 0
    total_masks = 0

    for _, row in selected.iterrows():

        image_name = str(
            row["image_name"]
        )

        mask_path = (
            SOURCE_ROOT
            / Path(image_name).with_suffix(".tif")
        )

        if not mask_path.exists():
            continue

        mask = np.array(
            Image.open(mask_path)
        )

        binary = (
            mask > 0
        ).astype(np.uint8)

        total_masks += 1

        pixels = int(
            binary.sum()
        )

        if pixels == 0:
            empty_masks += 1
            continue

        total_pixels = binary.shape[0] * binary.shape[1]

        mask_area.append(
            pixels / total_pixels
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        components = []

        for component_id in range(
            1,
            num_labels
        ):

            area = int(
                stats[
                    component_id,
                    cv2.CC_STAT_AREA
                ]
            )

            if area < MIN_COMPONENT_PIXELS:
                continue

            components.append(area)

        component_counts.append(
            len(components)
        )

        for area in components:
            component_area.append(
                area / total_pixels
            )

    report = {

        "total_masks": total_masks,

        "empty_masks": empty_masks,

        "mask_area_fraction": {
            "mean": float(
                np.mean(mask_area)
            ) if mask_area else 0.0,

            "median": float(
                np.median(mask_area)
            ) if mask_area else 0.0,

            "p10": float(
                np.percentile(
                    mask_area,
                    10
                )
            ) if mask_area else 0.0,

            "p90": float(
                np.percentile(
                    mask_area,
                    90
                )
            ) if mask_area else 0.0,
        },

        "components_per_mask": {
            "mean": float(
                np.mean(component_counts)
            ) if component_counts else 0.0,

            "median": float(
                np.median(component_counts)
            ) if component_counts else 0.0,

            "max": int(
                max(component_counts)
            ) if component_counts else 0,
        },

        "component_area_fraction": {
            "mean": float(
                np.mean(component_area)
            ) if component_area else 0.0,

            "median": float(
                np.median(component_area)
            ) if component_area else 0.0,

            "p10": float(
                np.percentile(
                    component_area,
                    10
                )
            ) if component_area else 0.0,

            "p90": float(
                np.percentile(
                    component_area,
                    90
                )
            ) if component_area else 0.0,
        },
    }

    print("=" * 70)
    print("MultiFire20K - Mask Structure Analysis")
    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=4
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()