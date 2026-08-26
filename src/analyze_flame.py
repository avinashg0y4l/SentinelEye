from pathlib import Path
from collections import Counter

ROOT = Path("data/processed/FLAME")

splits = {
    "Training": ROOT / "Training",
    "Test": ROOT / "Test",
}

for split_name, split_path in splits.items():
    print(f"\n{'=' * 50}")
    print(split_name)
    print(f"{'=' * 50}")

    for class_name in ["Fire", "No_Fire"]:
        folder = split_path / class_name

        files = [
            p for p in folder.iterdir()
            if p.is_file()
        ]

        extensions = Counter(
            p.suffix.lower()
            for p in files
        )

        print(f"\n{class_name}")
        print(f"Images: {len(files)}")
        print(f"Extensions: {dict(extensions)}")

    total = sum(
        len([
            p for p in (split_path / class_name).iterdir()
            if p.is_file()
        ])
        for class_name in ["Fire", "No_Fire"]
    )

    print(f"\nTotal: {total}")