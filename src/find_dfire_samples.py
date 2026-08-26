from pathlib import Path

ROOT = Path("data/processed/DFIRE/train")

fire_sample = None
smoke_sample = None
empty_sample = None

for label_path in sorted((ROOT / "labels").glob("*.txt")):

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    image_path = ROOT / "images" / f"{label_path.stem}.jpg"

    if not image_path.exists():
        continue

    if not text:
        if empty_sample is None:
            empty_sample = (image_path, label_path)

    else:
        class_ids = []

        for line in text.splitlines():
            parts = line.split()

            if len(parts) == 5:
                class_ids.append(int(parts[0]))

        if 0 in class_ids and fire_sample is None:
            fire_sample = (image_path, label_path)

        if 1 in class_ids and smoke_sample is None:
            smoke_sample = (image_path, label_path)

    if fire_sample and smoke_sample and empty_sample:
        break


print("\n=== FIRE SAMPLE ===")
print("Image:", fire_sample[0] if fire_sample else "Not found")
print("Label:", fire_sample[1] if fire_sample else "Not found")

print("\n=== SMOKE SAMPLE ===")
print("Image:", smoke_sample[0] if smoke_sample else "Not found")
print("Label:", smoke_sample[1] if smoke_sample else "Not found")

print("\n=== EMPTY SAMPLE ===")
print("Image:", empty_sample[0] if empty_sample else "Not found")
print("Label:", empty_sample[1] if empty_sample else "Not found")