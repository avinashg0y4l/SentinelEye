from pathlib import Path

from ultralytics import YOLO


BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL = (
    BASE_DIR
    / "models"
    / "sentineleye_v2a.pt"
)

IMAGE = (
    BASE_DIR
    / "demo"
    / "FramesV1_3900.jpg"
)

def main():
    print("=" * 70)
    print("SentinelEye - Single Image Prediction Diagnostic")
    print("=" * 70)

    print("Model:", MODEL.resolve())
    print("Image:", IMAGE.resolve())

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL.resolve()}"
        )

    if not IMAGE.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE.resolve()}"
        )

    model = YOLO(str(MODEL))

    print("\nClasses:")
    print(model.names)

    for conf in [0.25, 0.40, 0.50, 0.60, 0.70]:
        print("\n" + "-" * 70)
        print(f"CONFIDENCE = {conf}")

        result = model.predict(
            source=str(IMAGE),
            imgsz=640,
            conf=conf,
            device="cpu",
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("No detections.")
            continue

        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, (cls, score, box) in enumerate(
            zip(classes, scores, boxes),
            start=1,
        ):
            class_id = int(cls)
            name = model.names.get(
                class_id,
                str(class_id),
            )

            x1, y1, x2, y2 = box

            print(
                f"{i}. "
                f"{name.upper():6s} "
                f"conf={float(score):.3f} "
                f"box=({int(x1)}, {int(y1)}, "
                f"{int(x2)}, {int(y2)})"
            )


if __name__ == "__main__":
    main()