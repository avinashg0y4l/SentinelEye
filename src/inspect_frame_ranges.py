from pathlib import Path
import re

ROOT = Path("data/processed/FLAME")

folders = [
    ROOT / "Training" / "Fire",
    ROOT / "Training" / "No_Fire",
    ROOT / "Test" / "Fire",
    ROOT / "Test" / "No_Fire",
]


def extract_frame_number(filename):
    match = re.search(r"frame(\d+)", filename)
    return int(match.group(1)) if match else None


for folder in folders:
    numbers = []

    for path in folder.glob("*.jpg"):
        number = extract_frame_number(path.name)

        if number is not None:
            numbers.append(number)

    numbers.sort()

    print(f"\n{'=' * 60}")
    print(folder)
    print(f"{'=' * 60}")

    print("Images:", len(numbers))

    if numbers:
        print("Minimum frame:", min(numbers))
        print("Maximum frame:", max(numbers))
        print("First 20 frame numbers:", numbers[:20])
        print("Last 20 frame numbers:", numbers[-20:])