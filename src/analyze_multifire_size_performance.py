from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v1-2/weights/best.pt"
)

ROOT = Path(
    "data/experiments/detector_v1/"
    "multifire_urban_test/FU"
)

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )
    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_gt(label_path):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            cls, xc, yc, w, h = map(float, parts)
        except ValueError:
            continue

        if int(cls) != 1:
            continue

        if w <= 0 or h <= 0:
            continue

        boxes.append(
            {
                "box": [
                    xc - w / 2,
                    yc - h / 2,
                    xc + w / 2,
                    yc + h / 2,
                ],
                "area": w * h,
            }
        )

    return boxes


def size_bucket(area):
    if area < 0.01:
        return "small"

    if area < 0.10:
        return "medium"

    return "large"


def main():

    print("=" * 70)
    print("SentinelEye - MultiFire Size-wise Performance")
    print("=" * 70)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    model = YOLO(str(MODEL))

    stats = {
        "small": {"gt": 0, "matched": 0},
        "medium": {"gt": 0, "matched": 0},
        "large": {"gt": 0, "matched": 0},
    }

    images = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    print("Images:", len(images))

    for index, image_path in enumerate(
        images,
        start=1,
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

        gt = load_gt(label_path)

        if not gt:
            continue

        result = model.predict(
            source=image,
            imgsz=640,
            conf=CONF_THRESHOLD,
            device=0,
            verbose=False,
        )[0]

        predictions = []

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy.cpu().numpy()
            )

            classes = (
                result.boxes.cls.cpu().numpy()
            )

            for box, cls in zip(
                boxes,
                classes,
            ):

                # Fire class only.
                if int(cls) != 1:
                    continue

                x1, y1, x2, y2 = box

                height, width = image.shape[:2]

                predictions.append(
                    [
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                    ]
                )

        matched_predictions = set()

        for target in gt:

            bucket = size_bucket(
                target["area"]
            )

            stats[bucket]["gt"] += 1

            best_iou = 0.0
            best_index = None

            for pred_index, pred in enumerate(
                predictions
            ):

                if pred_index in matched_predictions:
                    continue

                score = iou(
                    target["box"],
                    pred,
                )

                if score > best_iou:
                    best_iou = score
                    best_index = pred_index

            if (
                best_index is not None
                and best_iou >= IOU_THRESHOLD
            ):
                stats[bucket]["matched"] += 1
                matched_predictions.add(
                    best_index
                )

        if index % 100 == 0:
            print(
                f"Processed: "
                f"{index}/{len(images)}"
            )

    report = {}

    for bucket, values in stats.items():

        gt = values["gt"]
        matched = values["matched"]

        recall = (
            matched / gt
            if gt
            else 0.0
        )

        report[bucket] = {
            "ground_truth": gt,
            "matched": matched,
            "missed": gt - matched,
            "recall": recall,
            "recall_percent": recall * 100,
        }

    print("\n" + "=" * 70)
    print("SIZE-WISE RESULTS")
    print("=" * 70)

    for bucket in [
        "small",
        "medium",
        "large",
    ]:

        r = report[bucket]

        print(
            f"{bucket.capitalize():8s} | "
            f"GT={r['ground_truth']:4d} | "
            f"Matched={r['matched']:4d} | "
            f"Missed={r['missed']:4d} | "
            f"Recall={r['recall_percent']:.2f}%"
        )

    output = (
        ROOT.parent
        / "size_performance.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()