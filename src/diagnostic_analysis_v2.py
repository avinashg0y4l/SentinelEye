from pathlib import Path
from collections import defaultdict
import json

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/detect/runs/detect/results/"
    "detector_v1-2/weights/best.pt"
)

TEST_ROOT = Path("data/processed/DFIRE/test")

OUTPUT_ROOT = Path(
    "results/detector_v1/diagnostics_v2"
)

TOP_K = 20

CONF_THRESHOLD = 0.25

# Strong match
GOOD_IOU = 0.50

# Enough overlap to decide that two boxes
# probably refer to the same physical region.
RELATED_IOU = 0.30

CLASS_NAMES = {
    0: "Smoke",
    1: "Fire",
}


def xywh_to_xyxy(x, y, w, h, width, height):
    xc = x * width
    yc = y * height

    bw = w * width
    bh = h * height

    return [
        xc - bw / 2,
        yc - bh / 2,
        xc + bw / 2,
        yc + bh / 2,
    ]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    intersection = iw * ih

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


def load_ground_truth(label_path, width, height):
    ground_truth = []

    if not label_path.exists():
        return ground_truth

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return ground_truth

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

        if not all(
            0.0 <= value <= 1.0
            for value in [x, y, w, h]
        ):
            continue

        if w <= 0 or h <= 0:
            continue

        ground_truth.append(
            {
                "class_id": class_id,
                "box": xywh_to_xyxy(
                    x,
                    y,
                    w,
                    h,
                    width,
                    height,
                ),
            }
        )

    return ground_truth


def draw_box(image, box, text, color):
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

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def add_header(image, text):
    header_height = 50

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], header_height),
        (30, 30, 30),
        -1,
    )

    cv2.putText(
        image,
        text,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_image(category, rank, image_path, image, info):
    folder = OUTPUT_ROOT / category

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = (
        f"{category} | "
        f"{image_path.name} | "
        f"{info}"
    )

    add_header(image, header)

    output_path = (
        folder
        / f"{rank:02d}_{image_path.name}"
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


def find_best_candidate(gt, predictions, allowed_indices):
    best_index = None
    best_iou = 0.0

    for pred_index in allowed_indices:

        pred = predictions[pred_index]

        score = calculate_iou(
            gt["box"],
            pred["box"],
        )

        if score > best_iou:
            best_iou = score
            best_index = pred_index

    return best_index, best_iou


def main():

    print("=" * 72)
    print("SentinelEye - Diagnostic Analysis V2")
    print("=" * 72)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    image_files = sorted(
        (TEST_ROOT / "images").glob("*.jpg")
    )

    print(
        f"Test images: {len(image_files)}"
    )

    statistics = {
        "correct": 0,
        "true_miss_fire": 0,
        "true_miss_smoke": 0,
        "wrong_class_fire_to_smoke": 0,
        "wrong_class_smoke_to_fire": 0,
        "localization": 0,
        "duplicate_prediction": 0,
        "false_positive": 0,
    }

    examples = defaultdict(list)

    for index, image_path in enumerate(
        image_files,
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

            for box, cls, conf in zip(
                boxes,
                classes,
                confidences,
            ):

                predictions.append(
                    {
                        "class_id": int(cls),
                        "box": box.tolist(),
                        "confidence": float(conf),
                    }
                )

        # --------------------------------------------------
        # STEP 1
        # Match exact same-class predictions
        # --------------------------------------------------

        unmatched_gt = set(
            range(len(ground_truth))
        )

        unmatched_pred = set(
            range(len(predictions))
        )

        exact_matches = []

        candidate_pairs = []

        for gt_index, gt in enumerate(
            ground_truth
        ):

            for pred_index, pred in enumerate(
                predictions
            ):

                if (
                    gt["class_id"]
                    != pred["class_id"]
                ):
                    continue

                score = calculate_iou(
                    gt["box"],
                    pred["box"],
                )

                if score >= GOOD_IOU:
                    candidate_pairs.append(
                        (
                            score,
                            gt_index,
                            pred_index,
                        )
                    )

        candidate_pairs.sort(
            reverse=True
        )

        for score, gt_index, pred_index in candidate_pairs:

            if gt_index not in unmatched_gt:
                continue

            if pred_index not in unmatched_pred:
                continue

            unmatched_gt.remove(gt_index)
            unmatched_pred.remove(pred_index)

            exact_matches.append(
                (
                    gt_index,
                    pred_index,
                    score,
                )
            )

        # --------------------------------------------------
        # STEP 2
        # Analyze unmatched GT objects
        # --------------------------------------------------

        for gt_index in list(
            unmatched_gt
        ):

            gt = ground_truth[gt_index]

            related_candidates = []

            for pred_index in unmatched_pred:

                pred = predictions[pred_index]

                score = calculate_iou(
                    gt["box"],
                    pred["box"],
                )

                if score >= RELATED_IOU:

                    related_candidates.append(
                        (
                            score,
                            pred_index,
                        )
                    )

            related_candidates.sort(
                reverse=True
            )

            if related_candidates:

                best_iou, pred_index = (
                    related_candidates[0]
                )

                pred = predictions[pred_index]

                # ------------------------------------------
                # Wrong class
                # ------------------------------------------

                if (
                    gt["class_id"]
                    != pred["class_id"]
                ):

                    if (
                        gt["class_id"] == 1
                        and pred["class_id"] == 0
                    ):
                        category = (
                            "04_wrong_class_fire_to_smoke"
                        )

                        statistics[
                            "wrong_class_fire_to_smoke"
                        ] += 1

                    elif (
                        gt["class_id"] == 0
                        and pred["class_id"] == 1
                    ):
                        category = (
                            "03_wrong_class_smoke_to_fire"
                        )

                        statistics[
                            "wrong_class_smoke_to_fire"
                        ] += 1

                    else:
                        continue

                    examples[category].append(
                        {
                            "image_path": image_path,
                            "ground_truth": ground_truth,
                            "predictions": predictions,
                            "info": (
                                f"GT={CLASS_NAMES[gt['class_id']]} "
                                f"Pred={CLASS_NAMES[pred['class_id']]} "
                                f"IoU={best_iou:.2f}"
                            ),
                        }
                    )

                    unmatched_gt.remove(
                        gt_index
                    )

                    unmatched_pred.remove(
                        pred_index
                    )

                    continue

                # ------------------------------------------
                # Same class but poor IoU
                # ------------------------------------------

                category = (
                    "05_localization"
                )

                statistics[
                    "localization"
                ] += 1

                examples[category].append(
                    {
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "predictions": predictions,
                        "info": (
                            f"Class={CLASS_NAMES[gt['class_id']]} "
                            f"IoU={best_iou:.2f}"
                        ),
                    }
                )

                unmatched_gt.remove(
                    gt_index
                )

                unmatched_pred.remove(
                    pred_index
                )

            else:

                # ------------------------------------------
                # True miss
                # ------------------------------------------

                if gt["class_id"] == 1:

                    category = "01_true_miss_fire"

                    statistics[
                        "true_miss_fire"
                    ] += 1

                else:

                    category = "02_true_miss_smoke"

                    statistics[
                        "true_miss_smoke"
                    ] += 1

                examples[category].append(
                    {
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "predictions": predictions,
                        "info": (
                            f"Missed "
                            f"{CLASS_NAMES[gt['class_id']]}"
                        ),
                    }
                )

                unmatched_gt.remove(
                    gt_index
                )

        # --------------------------------------------------
        # STEP 3
        # Remaining predictions
        # --------------------------------------------------

        for pred_index in list(
            unmatched_pred
        ):

            pred = predictions[pred_index]

            # Check whether prediction overlaps
            # a GT object already matched.
            duplicate_target = False

            for gt_index, matched_pred_index, _ in exact_matches:

                matched_gt = ground_truth[gt_index]

                score = calculate_iou(
                    matched_gt["box"],
                    pred["box"],
                )

                if (
                    matched_gt["class_id"]
                    == pred["class_id"]
                    and score >= RELATED_IOU
                ):

                    duplicate_target = True
                    break

            if duplicate_target:

                category = (
                    "06_duplicate_prediction"
                )

                statistics[
                    "duplicate_prediction"
                ] += 1

                examples[category].append(
                    {
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "predictions": predictions,
                        "info": (
                            f"Duplicate "
                            f"{CLASS_NAMES[pred['class_id']]} "
                            f"{pred['confidence']:.2f}"
                        ),
                    }
                )

            else:

                category = (
                    "07_false_positive"
                )

                statistics[
                    "false_positive"
                ] += 1

                examples[category].append(
                    {
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "predictions": predictions,
                        "info": (
                            f"False "
                            f"{CLASS_NAMES[pred['class_id']]} "
                            f"{pred['confidence']:.2f}"
                        ),
                    }
                )

        # --------------------------------------------------
        # Correct image/object accounting
        # --------------------------------------------------

        if (
            ground_truth
            and len(exact_matches) == len(ground_truth)
            and not unmatched_pred
        ):
            statistics["correct"] += 1

            category = "00_correct"

            examples[category].append(
                {
                    "image_path": image_path,
                    "ground_truth": ground_truth,
                    "predictions": predictions,
                    "info": "All GT objects matched",
                }
            )

        if index % 500 == 0:
            print(
                f"Processed: "
                f"{index}/{len(image_files)}"
            )

    # ------------------------------------------------------
    # SAVE REPRESENTATIVE IMAGES
    # ------------------------------------------------------

    for category, items in examples.items():

        # Most informative examples first:
        # higher number of boxes/errors first.
        items.sort(
            key=lambda item: (
                len(item["ground_truth"])
                + len(item["predictions"])
            ),
            reverse=True,
        )

        for rank, item in enumerate(
            items[:TOP_K],
            start=1,
        ):

            image = cv2.imread(
                str(item["image_path"])
            )

            if image is None:
                continue

            # Ground truth = GREEN
            for gt in item["ground_truth"]:

                draw_box(
                    image,
                    gt["box"],
                    f"GT: {CLASS_NAMES[gt['class_id']]}",
                    (0, 255, 0),
                )

            # Prediction = RED
            for pred in item["predictions"]:

                draw_box(
                    image,
                    pred["box"],
                    (
                        f"PRED: "
                        f"{CLASS_NAMES[pred['class_id']]} "
                        f"{pred['confidence']:.2f}"
                    ),
                    (0, 0, 255),
                )

            save_image(
                category,
                rank,
                item["image_path"],
                image,
                item["info"],
            )

    # ------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_ROOT / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            statistics,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("DIAGNOSTIC V2 COMPLETE")
    print("=" * 72)

    for key, value in statistics.items():
        print(
            f"{key:35s}: {value}"
        )

    print("\nSaved to:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()