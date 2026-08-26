from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_PATH = Path(
    "data/experiments/dfire_sample2/train/images/AoF00002.jpg"
)

LABEL_PATH = Path(
    "data/experiments/dfire_sample2/train/labels/AoF00002.txt"
)

OUTPUT_PATH = Path(
    "results/dfire_sample_AoF00002.jpg"
)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

width, height = image.size

print("Image size:", width, "x", height)


# --------------------------------------------------
# Load annotations
# --------------------------------------------------

lines = LABEL_PATH.read_text().strip().splitlines()

draw = ImageDraw.Draw(image)


for line in lines:

    if not line.strip():
        continue

    values = line.split()

    class_id = int(values[0])

    x_center = float(values[1])
    y_center = float(values[2])
    box_width = float(values[3])
    box_height = float(values[4])

    # Normalized → pixel coordinates

    x_center_px = x_center * width
    y_center_px = y_center * height

    box_width_px = box_width * width
    box_height_px = box_height * height

    x1 = int(
        x_center_px - box_width_px / 2
    )

    y1 = int(
        y_center_px - box_height_px / 2
    )

    x2 = int(
        x_center_px + box_width_px / 2
    )

    y2 = int(
        y_center_px + box_height_px / 2
    )

    # Draw bounding box

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )

    label = (
        "Fire"
        if class_id == 0
        else "Smoke"
    )

    draw.text(
        (x1, max(0, y1 - 18)),
        label
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

image.save(OUTPUT_PATH)

print("Saved:", OUTPUT_PATH)