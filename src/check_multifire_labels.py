from pathlib import Path
import random

import cv2


ROOT = Path(
    "data/experiments/detector_v1/"
    "multifire_urban_test"
)

IMAGE_DIR = ROOT / "FU" / "images"
LABEL_DIR = ROOT / "FU" / "labels"

OUTPUT = ROOT / "label_check"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAME = "Fire"

random.seed(42)


def draw_boxes(image, label_path):

    height, width = image.shape[:2]

    if not label_path.exists():
        return image

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, w, h = map(float, parts)

        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)
        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            CLASS_NAME,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


def main():

    images = list(
        IMAGE_DIR.glob("*.jpg")
    )

    print(
        "Total images:",
        len(images)
    )

    sample = random.sample(
        images,
        min(20, len(images))
    )

    for index, image_path in enumerate(
        sample,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        image = draw_boxes(
            image,
            label_path
        )

        output = (
            OUTPUT
            / f"{index:02d}_{image_path.name}"
        )

        cv2.imwrite(
            str(output),
            image
        )

    print(
        "Saved:",
        OUTPUT
    )


if __name__ == "__main__":
    main()