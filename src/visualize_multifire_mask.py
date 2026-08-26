from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(
    "data/experiments/multifire_sample/Test/FU"
)

IMAGE_PATH = ROOT / "FramesV1_1020.jpg"
MASK_PATH = ROOT / "FramesV1_1020.tif"
OUTPUT_PATH = Path(
    "results/multifire_mask_overlay/FramesV1_1020_overlay.jpg"
)


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("L")

    image_array = np.array(image)
    mask_array = np.array(mask)

    # Resize mask to the original image resolution.
    mask_resized = Image.fromarray(mask_array).resize(
        image.size,
        resample=Image.Resampling.NEAREST,
    )

    mask_array = np.array(mask_resized)

    overlay = image_array.copy()

    # Highlight mask==1.
    fire_pixels = mask_array == 1

    # Make the mask visible by blending.
    overlay[fire_pixels] = (
        0.5 * overlay[fire_pixels]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(overlay).save(OUTPUT_PATH)

    print("Saved:", OUTPUT_PATH)
    print("Image size:", image.size)
    print("Mask size:", mask.size)
    print("Mask pixels:", int(fire_pixels.sum()))


if __name__ == "__main__":
    main()