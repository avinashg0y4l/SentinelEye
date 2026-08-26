from pathlib import Path
from PIL import Image
import random

DATASET = Path("data/processed/FLAME")

samples = {
    "Training Fire": DATASET / "Training" / "Fire",
    "Training No_Fire": DATASET / "Training" / "No_Fire",
    "Test Fire": DATASET / "Test" / "Fire",
    "Test No_Fire": DATASET / "Test" / "No_Fire",
}

for name, folder in samples.items():
    images = list(folder.glob("*.jpg"))

    selected = random.choice(images)

    print(f"{name}:")
    print(f"  {selected}")
    print(f"  Size: {Image.open(selected).size}")