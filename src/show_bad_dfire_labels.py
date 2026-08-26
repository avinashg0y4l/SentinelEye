from pathlib import Path

TRAIN_FILES = [
    "AoF05470.txt",
    "AoF06155.txt",
    "AoF06348.txt",
    "AoF06439.txt",
    "AoF06456.txt",
]

TEST_FILES = [
    "AoF07743.txt",
    "AoF07774.txt",
    "AoF08348.txt",
    "WEB10669.txt",
    "WEB10769.txt",
]

train_root = Path("data/processed/DFIRE/train/labels")
test_root = Path("data/processed/DFIRE/test/labels")

print("=" * 60)
print("TRAIN BAD ANNOTATIONS")
print("=" * 60)

for filename in TRAIN_FILES:
    path = train_root / filename

    print(f"\n--- {filename} ---")

    if path.exists():
        print(path.read_text(
            encoding="utf-8",
            errors="ignore"
        ))
    else:
        print("FILE NOT FOUND")


print("\n" + "=" * 60)
print("TEST BAD ANNOTATIONS")
print("=" * 60)

for filename in TEST_FILES:
    path = test_root / filename

    print(f"\n--- {filename} ---")

    if path.exists():
        print(path.read_text(
            encoding="utf-8",
            errors="ignore"
        ))
    else:
        print("FILE NOT FOUND")