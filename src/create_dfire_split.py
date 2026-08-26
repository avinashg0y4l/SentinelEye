from pathlib import Path
import random
import shutil
import json

SOURCE = Path("data/experiments/detector_v1/DFIRE_clean/train")
OUTPUT = Path("data/experiments/detector_v1/DFIRE_clean_split")

VAL_RATIO = 0.20
SEED = 42

IMAGE_DIR = SOURCE / "images"
LABEL_DIR = SOURCE / "labels"


def main():
    random.seed(SEED)

    pairs = []

    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if label_path.exists():
            pairs.append((image_path, label_path))

    random.shuffle(pairs)

    val_count = int(len(pairs) * VAL_RATIO)

    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print("Total pairs:", len(pairs))
    print("Train pairs:", len(train_pairs))
    print("Val pairs:", len(val_pairs))

    for split in ["train", "val"]:
        (OUTPUT / split / "images").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT / split / "labels").mkdir(
            parents=True,
            exist_ok=True
        )

    for image_path, label_path in train_pairs:
        shutil.copy2(
            image_path,
            OUTPUT / "train" / "images" / image_path.name
        )
        shutil.copy2(
            label_path,
            OUTPUT / "train" / "labels" / label_path.name
        )

    for image_path, label_path in val_pairs:
        shutil.copy2(
            image_path,
            OUTPUT / "val" / "images" / image_path.name
        )
        shutil.copy2(
            label_path,
            OUTPUT / "val" / "labels" / label_path.name
        )

    metadata = {
        "seed": SEED,
        "validation_ratio": VAL_RATIO,
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
    }

    (OUTPUT / "split_metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\nSplit created:")
    print(OUTPUT)


if __name__ == "__main__":
    main()