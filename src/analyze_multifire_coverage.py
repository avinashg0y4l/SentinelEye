from pathlib import Path
import json

import cv2
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


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0.0, box1[2] - box1[0])
        * max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0])
        * max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def prediction_inside_gt_fraction(gt, pred):
    """
    What fraction of the prediction area lies inside GT?

    intersection / prediction_area
    """
    x1 = max(gt[0], pred[0])
    y1 = max(gt[1], pred[1])
    x2 = min(gt[2], pred[2])
    y2 = min(gt[3], pred[3])

    intersection_width = max(
        0.0,
        x2 - x1
    )

    intersection_height = max(
        0.0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    prediction_area = (
        max(0.0, pred[2] - pred[0])
        * max(0.0, pred[3] - pred[1])
    )

    if prediction_area <= 0:
        return 0.0

    return intersection / prediction_area


def gt_coverage(gt, pred):
    """
    What fraction of the GT area is covered by prediction?

    intersection / GT_area
    """
    x1 = max(gt[0], pred[0])
    y1 = max(gt[1], pred[1])
    x2 = min(gt[2], pred[2])
    y2 = min(gt[3], pred[3])

    intersection_width = max(
        0.0,
        x2 - x1
    )

    intersection_height = max(
        0.0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    gt_area = (
        max(0.0, gt[2] - gt[0])
        * max(0.0, gt[3] - gt[1])
    )

    if gt_area <= 0:
        return 0.0

    return intersection / gt_area


def load_gt(label_path):
    """
    Load Fire ground-truth boxes from YOLO labels.

    Class convention:
        0 = Smoke
        1 = Fire

    MultiFire test labels contain Fire only.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            continue

        # Fire only
        if class_id != 1:
            continue

        if w <= 0 or h <= 0:
            continue

        boxes.append({
            "box": [
                xc - w / 2,
                yc - h / 2,
                xc + w / 2,
                yc + h / 2,
            ],
            "area": w * h,
        })

    return boxes


def size_bucket(area):
    """
    Area is normalized by image area.

    small  < 1%
    medium 1% - 10%
    large  >= 10%
    """
    if area < 0.01:
        return "small"

    if area < 0.10:
        return "medium"

    return "large"


def main():

    print("=" * 70)
    print("SentinelEye - MultiFire Prediction Coverage")
    print("=" * 70)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found: {LABEL_DIR}"
        )

    model = YOLO(str(MODEL))

    stats = {
        "small": {
            "gt": 0,

            "no_prediction": 0,
            "prediction_nearby": 0,

            "good_iou": 0,

            "gt_coverage_30": 0,
            "gt_coverage_50": 0,
            "gt_coverage_70": 0,

            "prediction_inside_30": 0,
            "prediction_inside_50": 0,
        },

        "medium": {
            "gt": 0,

            "no_prediction": 0,
            "prediction_nearby": 0,

            "good_iou": 0,

            "gt_coverage_30": 0,
            "gt_coverage_50": 0,
            "gt_coverage_70": 0,

            "prediction_inside_30": 0,
            "prediction_inside_50": 0,
        },

        "large": {
            "gt": 0,

            "no_prediction": 0,
            "prediction_nearby": 0,

            "good_iou": 0,

            "gt_coverage_30": 0,
            "gt_coverage_50": 0,
            "gt_coverage_70": 0,

            "prediction_inside_30": 0,
            "prediction_inside_50": 0,
        },
    }

    images = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    print("Images:", len(images))

    for index, image_path in enumerate(
        images,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "Warning: could not read:",
                image_path
            )
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        ground_truth = load_gt(
            label_path
        )

        if not ground_truth:
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
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for box, class_id, confidence in zip(
                boxes,
                classes,
                confidences,
            ):

                # Fire only
                if int(class_id) != 1:
                    continue

                x1, y1, x2, y2 = box

                predictions.append({
                    "box": [
                        x1 / image_width,
                        y1 / image_height,
                        x2 / image_width,
                        y2 / image_height,
                    ],
                    "confidence": float(
                        confidence
                    ),
                })

        # -----------------------------------------------
        # Analyse every GT box
        # -----------------------------------------------

        for gt in ground_truth:

            gt_box = gt["box"]
            area = gt["area"]

            bucket = size_bucket(area)

            stats[bucket]["gt"] += 1

            best_iou_score = 0.0
            best_gt_coverage = 0.0
            best_prediction_inside = 0.0

            for prediction in predictions:

                prediction_box = prediction["box"]

                iou_score = calculate_iou(
                    gt_box,
                    prediction_box,
                )

                gt_coverage_score = gt_coverage(
                    gt_box,
                    prediction_box,
                )

                prediction_inside_score = (
                    prediction_inside_gt_fraction(
                        gt_box,
                        prediction_box,
                    )
                )

                best_iou_score = max(
                    best_iou_score,
                    iou_score,
                )

                best_gt_coverage = max(
                    best_gt_coverage,
                    gt_coverage_score,
                )

                best_prediction_inside = max(
                    best_prediction_inside,
                    prediction_inside_score,
                )

            # -------------------------------------------
            # Nearby prediction
            #
            # At least 30% of prediction area
            # lies within the GT region.
            # -------------------------------------------

            if best_prediction_inside >= 0.30:
                stats[bucket][
                    "prediction_nearby"
                ] += 1

            # -------------------------------------------
            # No meaningful prediction
            # -------------------------------------------

            if best_prediction_inside < 0.30:
                stats[bucket][
                    "no_prediction"
                ] += 1

            # -------------------------------------------
            # Standard IoU
            # -------------------------------------------

            if best_iou_score >= 0.50:
                stats[bucket][
                    "good_iou"
                ] += 1

            # -------------------------------------------
            # GT coverage
            # -------------------------------------------

            if best_gt_coverage >= 0.30:
                stats[bucket][
                    "gt_coverage_30"
                ] += 1

            if best_gt_coverage >= 0.50:
                stats[bucket][
                    "gt_coverage_50"
                ] += 1

            if best_gt_coverage >= 0.70:
                stats[bucket][
                    "gt_coverage_70"
                ] += 1

            # -------------------------------------------
            # Prediction containment
            # -------------------------------------------

            if best_prediction_inside >= 0.30:
                stats[bucket][
                    "prediction_inside_30"
                ] += 1

            if best_prediction_inside >= 0.50:
                stats[bucket][
                    "prediction_inside_50"
                ] += 1

        if index % 100 == 0:
            print(
                f"Processed: "
                f"{index}/{len(images)}"
            )

    # ===================================================
    # FINAL REPORT
    # ===================================================

    print("\n" + "=" * 70)
    print("COVERAGE RESULTS")
    print("=" * 70)

    report = {}

    for bucket in [
        "small",
        "medium",
        "large",
    ]:

        bucket_stats = stats[bucket]

        gt_count = bucket_stats["gt"]

        if gt_count == 0:
            continue

        nearby_percent = (
            100
            * bucket_stats["prediction_nearby"]
            / gt_count
        )

        no_prediction_percent = (
            100
            * bucket_stats["no_prediction"]
            / gt_count
        )

        iou_percent = (
            100
            * bucket_stats["good_iou"]
            / gt_count
        )

        gt_coverage_30_percent = (
            100
            * bucket_stats["gt_coverage_30"]
            / gt_count
        )

        gt_coverage_50_percent = (
            100
            * bucket_stats["gt_coverage_50"]
            / gt_count
        )

        gt_coverage_70_percent = (
            100
            * bucket_stats["gt_coverage_70"]
            / gt_count
        )

        prediction_inside_30_percent = (
            100
            * bucket_stats["prediction_inside_30"]
            / gt_count
        )

        prediction_inside_50_percent = (
            100
            * bucket_stats["prediction_inside_50"]
            / gt_count
        )

        print(
            f"\n{bucket.upper()}"
        )

        print(
            f"GT boxes                 : {gt_count}"
        )

        print(
            f"Nearby prediction       : "
            f"{nearby_percent:6.2f}%"
        )

        print(
            f"No prediction           : "
            f"{no_prediction_percent:6.2f}%"
        )

        print(
            f"IoU >= 0.50             : "
            f"{iou_percent:6.2f}%"
        )

        print(
            f"GT coverage >= 30%      : "
            f"{gt_coverage_30_percent:6.2f}%"
        )

        print(
            f"GT coverage >= 50%      : "
            f"{gt_coverage_50_percent:6.2f}%"
        )

        print(
            f"GT coverage >= 70%      : "
            f"{gt_coverage_70_percent:6.2f}%"
        )

        print(
            f"Prediction inside >=30% : "
            f"{prediction_inside_30_percent:6.2f}%"
        )

        print(
            f"Prediction inside >=50% : "
            f"{prediction_inside_50_percent:6.2f}%"
        )

        report[bucket] = {
            "ground_truth": gt_count,

            "nearby_prediction_percent":
                nearby_percent,

            "no_prediction_percent":
                no_prediction_percent,

            "good_iou_percent":
                iou_percent,

            "gt_coverage_30_percent":
                gt_coverage_30_percent,

            "gt_coverage_50_percent":
                gt_coverage_50_percent,

            "gt_coverage_70_percent":
                gt_coverage_70_percent,

            "prediction_inside_30_percent":
                prediction_inside_30_percent,

            "prediction_inside_50_percent":
                prediction_inside_50_percent,
        }

    # ===================================================
    # SAVE REPORT
    # ===================================================

    output = (
        ROOT.parent
        / "coverage_analysis.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Saved:")
    print(output)
    print("=" * 70)


if __name__ == "__main__":
    main()