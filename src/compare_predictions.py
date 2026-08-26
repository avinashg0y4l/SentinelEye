from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v1-2/weights/best.pt"
)

TEST_ROOT = Path("data/processed/DFIRE/test")

OUTPUT = Path(
    "results/detector_v1/prediction_vs_ground_truth"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAMES = {
    0: "Smoke",
    1: "Fire",
}

# Green = ground truth
GT_COLOR = (0, 255, 0)

# Red = prediction
PRED_COLOR = (0, 0, 255)

CONF_THRESHOLD = 0.25


def load_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return boxes

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        if w <= 0 or h <= 0:
            continue

        x_center = x * width
        y_center = y * height

        box_width = w * width
        box_height = h * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        boxes.append(
            (
                class_id,
                [x1, y1, x2, y2]
            )
        )

    return boxes


def draw_box(
    image,
    box,
    text,
    color,
):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():

    print("=" * 60)
    print("SentinelEye - Prediction vs Ground Truth")
    print("=" * 60)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    model = YOLO(str(MODEL))

    image_files = sorted(
        TEST_ROOT.joinpath("images").glob("*.jpg")
    )

    print("Test images:", len(image_files))

    # We only need a representative sample first.
    sample_images = image_files[:30]

    for index, image_path in enumerate(
        sample_images,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = (
            TEST_ROOT
            / "labels"
            / f"{image_path.stem}.txt"
        )

        ground_truth = load_ground_truth(
            label_path,
            width,
            height,
        )

        # Draw ground truth in GREEN
        for class_id, box in ground_truth:

            draw_box(
                image,
                box,
                f"GT: {CLASS_NAMES[class_id]}",
                GT_COLOR,
            )

        # Run prediction
        result = model.predict(
            source=image,
            imgsz=640,
            conf=CONF_THRESHOLD,
            device=0,
            verbose=False,
        )[0]

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(
                boxes,
                classes,
                confidences,
            ):

                box = [int(v) for v in box]

                class_id = int(cls)

                draw_box(
                    image,
                    box,
                    f"PRED: {CLASS_NAMES[class_id]} {conf:.2f}",
                    PRED_COLOR,
                )

        output_path = (
            OUTPUT
            / f"{index:03d}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        print(
            f"{index:02d}/{len(sample_images)} "
            f"{image_path.name}"
        )

    print("\nSaved comparison images:")
    print(OUTPUT)


if __name__ == "__main__":
    main()