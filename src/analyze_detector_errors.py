from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/detect/runs/detect/results/"
    "detector_v1-2/weights/best.pt"
)

TEST_ROOT = Path(
    "data/processed/DFIRE/test"
)

OUTPUT_ROOT = Path(
    "results/detector_v1/error_analysis"
)

FN_DIR = OUTPUT_ROOT / "false_negatives"
FP_DIR = OUTPUT_ROOT / "false_positives"

FN_DIR.mkdir(parents=True, exist_ok=True)
FP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

MAX_SAVED = 50

CLASS_NAMES = {
    0: "Smoke",
    1: "Fire",
}

def xywh_to_xyxy(x, y, w, h, width, height):
    x_center = x * width
    y_center = y * height

    box_width = w * width
    box_height = h * height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def read_ground_truth(label_path, width, height):

    ground_truth = []
    invalid_rows = []

    if not label_path.exists():
        return ground_truth, invalid_rows

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return ground_truth, invalid_rows

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        parts = line.split()

        if len(parts) != 5:
            invalid_rows.append(line_number)
            continue

        try:
            class_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            invalid_rows.append(line_number)
            continue

        if class_id not in CLASS_NAMES:
            invalid_rows.append(line_number)
            continue

        if not all(
            0.0 <= value <= 1.0
            for value in [x, y, w, h]
        ):
            invalid_rows.append(line_number)
            continue

        if w <= 0 or h <= 0:
            invalid_rows.append(line_number)
            continue

        ground_truth.append({
            "class_id": class_id,
            "box": xywh_to_xyxy(
                x,
                y,
                w,
                h,
                width,
                height,
            ),
        })

    return ground_truth, invalid_rows


def draw_box(
    image,
    box,
    label,
    confidence=None,
    color=(0, 0, 255),
):
    x1, y1, x2, y2 = [
        int(v) for v in box
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    if confidence is not None:
        text = f"{label} {confidence:.2f}"
    else:
        text = label

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():

    print("=" * 60)
    print("SentinelEye - Detector Error Analysis")
    print("=" * 60)

    print("Model:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    image_files = sorted(
        list((TEST_ROOT / "images").glob("*.jpg"))
        +
        list((TEST_ROOT / "images").glob("*.jpeg"))
        +
        list((TEST_ROOT / "images").glob("*.png"))
    )

    print("Test images found:", len(image_files))

    total_images = 0
    valid_images = 0

    total_gt = 0
    total_predictions = 0

    false_negative_objects = 0
    false_positive_objects = 0

    false_negative_images = defaultdict(list)
    false_positive_images = defaultdict(list)

    for index, image_path in enumerate(image_files):

        total_images += 1

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

        ground_truth, invalid_rows = read_ground_truth(
            label_path,
            width,
            height,
        )

        # Skip images where the only information available
        # is malformed annotation.
        if invalid_rows and not ground_truth:
            continue

        valid_images += 1

        total_gt += len(ground_truth)

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=0.50,
            device=0,
            verbose=False,
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(
                boxes,
                classes,
                confidences,
            ):

                predictions.append({
                    "class_id": int(cls),
                    "box": box.tolist(),
                    "confidence": float(conf),
                })

        total_predictions += len(predictions)

        # --------------------------------------------------
        # Match predictions with ground truth
        # --------------------------------------------------

        matched_gt = set()
        matched_predictions = set()

        candidates = []

        for gt_index, gt in enumerate(ground_truth):

            for pred_index, pred in enumerate(predictions):

                if (
                    gt["class_id"]
                    != pred["class_id"]
                ):
                    continue

                iou = calculate_iou(
                    gt["box"],
                    pred["box"],
                )

                if iou >= IOU_THRESHOLD:
                    candidates.append(
                        (
                            iou,
                            gt_index,
                            pred_index,
                        )
                    )

        candidates.sort(
            reverse=True
        )

        for iou, gt_index, pred_index in candidates:

            if gt_index in matched_gt:
                continue

            if pred_index in matched_predictions:
                continue

            matched_gt.add(gt_index)
            matched_predictions.add(pred_index)

        # --------------------------------------------------
        # False negatives
        # --------------------------------------------------

        for gt_index, gt in enumerate(
            ground_truth
        ):

            if gt_index in matched_gt:
                continue

            false_negative_objects += 1

            class_name = CLASS_NAMES[
                gt["class_id"]
            ]

            false_negative_images[
                image_path
            ].append({
                "class_name": class_name,
                "box": gt["box"],
            })

        # --------------------------------------------------
        # False positives
        # --------------------------------------------------

        for pred_index, pred in enumerate(
            predictions
        ):

            if pred_index in matched_predictions:
                continue

            false_positive_objects += 1

            class_name = CLASS_NAMES[
                pred["class_id"]
            ]

            false_positive_images[
                image_path
            ].append({
                "class_name": class_name,
                "box": pred["box"],
                "confidence": pred["confidence"],
            })

        if (index + 1) % 500 == 0:
            print(
                f"Processed: "
                f"{index + 1}/{len(image_files)}"
            )

    # ------------------------------------------------------
    # Save representative false negatives
    # ------------------------------------------------------

    fn_ranked = sorted(
        false_negative_images.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    saved_fn = 0

    for image_path, errors in fn_ranked:

        if saved_fn >= MAX_SAVED:
            break

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # Draw missed ground-truth objects
        for error in errors:

            draw_box(
                image,
                error["box"],
                f"MISSED {error['class_name']}",
                color=(0, 0, 255),
            )

        output_path = (
            FN_DIR
            / f"{saved_fn + 1:03d}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        saved_fn += 1

    # ------------------------------------------------------
    # Save representative false positives
    # ------------------------------------------------------

    fp_ranked = sorted(
        false_positive_images.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    saved_fp = 0

    for image_path, errors in fp_ranked:

        if saved_fp >= MAX_SAVED:
            break

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # Draw false predictions
        for error in errors:

            draw_box(
                image,
                error["box"],
                f"FALSE {error['class_name']}",
                confidence=error["confidence"],
                color=(255, 0, 255),
            )

        output_path = (
            FP_DIR
            / f"{saved_fp + 1:03d}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        saved_fp += 1

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)

    print("Total images:", total_images)
    print("Usable images:", valid_images)

    print("Ground-truth objects:", total_gt)
    print("Predicted objects:", total_predictions)

    print(
        "False-negative objects:",
        false_negative_objects,
    )

    print(
        "False-positive objects:",
        false_positive_objects,
    )

    print(
        "Images with false negatives:",
        len(false_negative_images),
    )

    print(
        "Images with false positives:",
        len(false_positive_images),
    )

    print("\nSaved false negatives:")
    print(FN_DIR)

    print("\nSaved false positives:")
    print(FP_DIR)


if __name__ == "__main__":
    main()