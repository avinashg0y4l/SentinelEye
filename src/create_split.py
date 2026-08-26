from pathlib import Path
import random
import shutil

# -----------------------------
# Configuration
# -----------------------------

SOURCE = Path("data/processed/FLAME/Training")
OUTPUT = Path("data/experiments/baseline_v1")

CLASSES = ["Fire", "No_Fire"]

VALIDATION_RATIO = 0.20
SEED = 42

# -----------------------------
# Reproducibility
# -----------------------------

random.seed(SEED)

# -----------------------------
# Create output directories
# -----------------------------

for split in ["train", "val"]:
    for class_name in CLASSES:
        (OUTPUT / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# -----------------------------
# Create split
# -----------------------------

for class_name in CLASSES:

    source_dir = SOURCE / class_name

    images = sorted(
        [
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".jpg"
        ]
    )

    random.shuffle(images)

    validation_count = int(
        len(images) * VALIDATION_RATIO
    )

    val_images = images[:validation_count]
    train_images = images[validation_count:]

    print(f"\n{class_name}")
    print(f"Total: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")

    # Copy training images
    for image in train_images:
        destination = OUTPUT / "train" / class_name / image.name
        shutil.copy2(image, destination)

    # Copy validation images
    for image in val_images:
        destination = OUTPUT / "val" / class_name / image.name
        shutil.copy2(image, destination)

print("\nSplit creation completed.")
print(f"Output: {OUTPUT}")
print(f"Random seed: {SEED}")
print(f"Validation ratio: {VALIDATION_RATIO}")