from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("data/processed/DFIRE/train")
OUTPUT = Path("results/dfire_samples")
OUTPUT.mkdir(parents=True, exist_ok=True)

samples = {
    "fire": "AoF00002",
    "smoke": "AoF00001",
    "empty": "AoF00000",
}


def draw_annotations(image_path, label_path, output_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    draw = ImageDraw.Draw(image)

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        draw.text((10, 10), "NO OBJECT", fill="red")
        image.save(output_path)
        return

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        label = {
            0: "FIRE",
            1: "SMOKE",
        }.get(class_id, f"CLASS_{class_id}")

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4,
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            label,
            fill="red",
        )

    image.save(output_path)


for name, stem in samples.items():

    image_path = ROOT / "images" / f"{stem}.jpg"
    label_path = ROOT / "labels" / f"{stem}.txt"

    output_path = OUTPUT / f"{name}_{stem}.jpg"

    draw_annotations(
        image_path,
        label_path,
        output_path,
    )

    print("Saved:", output_path)