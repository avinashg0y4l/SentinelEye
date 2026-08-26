from pathlib import Path
from ultralytics import YOLO


CHECKPOINT = Path(
    "runs/detect/results/detector_v1/dfire_yolo11n/weights/last.pt"
)

DATA = (
    "data/experiments/detector_v1/"
    "DFIRE_clean_split/data.yaml"
)


def main():
    print("=" * 60)
    print("SentinelEye - Resume D-Fire Detector")
    print("=" * 60)

    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint exists:", CHECKPOINT.exists())

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    print("\nLoading checkpoint...")

    model = YOLO(str(CHECKPOINT))

    print("Checkpoint loaded successfully.")

    print("\nResuming training...")

    model.train(
        resume=True,
        data=DATA,
        device=0,
        workers=0,
        project="runs/detect/results/detector_v1",
        name="dfire_yolo11n",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()