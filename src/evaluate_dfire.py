from pathlib import Path

from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v2/dfire_multifire_manual/"
    "weights/best.pt"
)

DATA = "data/processed/DFIRE/test.yaml"


def main():
    print("=" * 60)
    print("SentinelEye - Official D-Fire Test - V2-A")
    print("=" * 60)

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
        project="runs/detect/results/detector_v2",
        name="dfire_official_test",
        exist_ok=True,
    )

    print("\nOfficial D-Fire V2-A test completed.")
    print("\nResults:")
    print(results)


if __name__ == "__main__":
    main()