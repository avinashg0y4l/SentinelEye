from pathlib import Path

from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v1-2/weights/best.pt"
)

DATA = (
    "data/experiments/detector_v1/"
    "multifire_urban_test/data.yaml"
)


def main():
    print("=" * 70)
    print("SentinelEye - MultiFire20K Urban Fire Evaluation")
    print("=" * 70)

    print("Model:", MODEL)
    print("Model exists:", MODEL.exists())

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    model = YOLO(str(MODEL))

    results = model.val(
        data=DATA,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project="runs/detect/results",
        name="multifire_urban_fire_test",
        exist_ok=True,
    )

    print("\nEvaluation completed.")
    print(results)


if __name__ == "__main__":
    main()