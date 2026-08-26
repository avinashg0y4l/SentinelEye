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
    "results/detector_v1/diagnostics"
)

TOP_K = 20

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

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


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    intersection = iw * ih

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_gt(label_path, width, height):
    gt = []

    if not label_path.exists():
        return gt

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return gt

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue

        if cls not in CLASS_NAMES:
            continue

        if not all(
            0 <= value <= 1
            for value in [x, y, w, h]
        ):
            continue

        if w <= 0 or h <= 0:
            continue

        gt.append({
            "class_id": cls,
            "box": xywh_to_xyxy(
                x, y, w, h, width, height
            )
        })

    return gt


def draw_box(
    image,
    box,
    text,
    color,
):
    x1, y1, x2, y2 = map(int, box)

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
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def save_diagnostic(
    category,
    image_path,
    image,
    gt_count,
    pred_count,
    extra_text,
    rank,
):
    folder = OUTPUT_ROOT / category
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    text = (
        f"Category: {category} | "
        f"GT: {gt_count} | "
        f"Pred: {pred_count} | "
        f"{extra_text}"
    )

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 38),
        (30, 30, 30),
        -1,
    )

    cv2.putText(
        image,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output = folder / f"{rank:02d}_{image_path.name}"

    cv2.imwrite(
        str(output),
        image
    )


def main():

    print("=" * 70)
    print("SentinelEye - Diagnostic Error Analysis")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    image_files = sorted(
        (TEST_ROOT / "images").glob("*.jpg")
    )

    print("Test images:", len(image_files))

    categories = defaultdict(list)

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = (
            TEST_ROOT
            / "labels"
            / f"{image_path.stem}.txt"
        )

        ground_truth = load_gt(
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

        matched_gt = set()
        matched_pred = set()
        matches = []

        # Find all valid same-class matches
        for gi, gt in enumerate(ground_truth):

            for pi, pred in enumerate(predictions):

                if gt["class_id"] != pred["class_id"]:
                    continue

                score = iou(
                    gt["box"],
                    pred["box"]
                )

                if score >= IOU_THRESHOLD:
                    matches.append(
                        (
                            score,
                            gi,
                            pi,
                        )
                    )

        # Best IoU first
        matches.sort(
            reverse=True
        )

        for score, gi, pi in matches:

            if gi in matched_gt:
                continue

            if pi in matched_pred:
                continue

            matched_gt.add(gi)
            matched_pred.add(pi)

        missed_gt = [
            gt
            for i, gt in enumerate(ground_truth)
            if i not in matched_gt
        ]

        unmatched_pred = [
            pred
            for i, pred in enumerate(predictions)
            if i not in matched_pred
        ]

        # --------------------------------------------
        # CATEGORY
        # --------------------------------------------

        if ground_truth and not predictions:

            missed_classes = {
                CLASS_NAMES[x["class_id"]]
                for x in missed_gt
            }

            if "Fire" in missed_classes:
                category = "02_missed_fire"
            elif "Smoke" in missed_classes:
                category = "03_missed_smoke"
            else:
                continue

        elif not ground_truth and predictions:

            category = "04_false_positive"

        elif ground_truth and predictions and missed_gt:

            missed_classes = {
                CLASS_NAMES[x["class_id"]]
                for x in missed_gt
            }

            if "Fire" in missed_classes:
                category = "02_missed_fire"
            elif "Smoke" in missed_classes:
                category = "03_missed_smoke"
            else:
                category = "05_localization"

        elif ground_truth and predictions and unmatched_pred:

            category = "05_localization"

        elif ground_truth and predictions:

            category = "01_correct"

        else:
            category = "01_correct"

        categories[category].append({
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "matched_gt": matched_gt,
            "matched_pred": matched_pred,
            "missed_gt": missed_gt,
            "unmatched_pred": unmatched_pred,
        })

        if index % 500 == 0:
            print(
                f"Processed: {index}/{len(image_files)}"
            )

    # --------------------------------------------
    # SAVE REPRESENTATIVE IMAGES
    # --------------------------------------------

    for category, items in categories.items():

        # prioritize examples with more errors
        items.sort(
            key=lambda x: (
                len(x["missed_gt"])
                + len(x["unmatched_pred"])
            ),
            reverse=True,
        )

        for rank, item in enumerate(
            items[:TOP_K],
            start=1
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

            save_diagnostic(
                category,
                item["image_path"],
                image,
                len(item["ground_truth"]),
                len(item["predictions"]),
                (
                    f"Missed={len(item['missed_gt'])} "
                    f"UnmatchedPred={len(item['unmatched_pred'])}"
                ),
                rank,
            )

    summary = {
        category: len(items)
        for category, items in categories.items()
    }

    (OUTPUT_ROOT).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    for category in sorted(summary):
        print(
            f"{category}: "
            f"{summary[category]}"
        )

    print("\nSaved to:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()