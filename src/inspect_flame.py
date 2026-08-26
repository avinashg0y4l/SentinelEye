from pathlib import Path
from PIL import Image

DATASET = Path("data/processed/FLAME")

folders = [
    DATASET / "Training" / "Fire",
    DATASET / "Training" / "No_Fire",
    DATASET / "Test" / "Fire",
    DATASET / "Test" / "No_Fire",
]

for folder in folders:
    print(f"\n=== {folder} ===")

    images = list(folder.glob("*.jpg"))

    print("Images:", len(images))

    if not images:
        continue

    resolutions = {}

    for image_path in images[:100]:
        try:
            with Image.open(image_path) as img:
                resolution = img.size
                resolutions[resolution] = resolutions.get(resolution, 0) + 1
        except Exception as e:
            print("Error:", image_path, e)

    print("Sample resolutions:")
    for resolution, count in resolutions.items():
        print(f"  {resolution}: {count}")