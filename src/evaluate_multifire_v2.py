from pathlib import Path

from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v2/dfire_multifire_manual/"
    "weights/best.pt"
)

DATA = (
    "data/experiments/detector_v1/"
    "multifire_urban_test/data.yaml"
)

PROJECT = "runs/detect/results/detector_v2"
NAME = "multifire_urban_fire_test"


def main():
    print("=" * 70)
    print("SentinelEye - V2-A MultiFire Urban Fire Object Evaluation")
    print("=" * 70)

    print("Model:", MODEL)
    print("Model exists:", MODEL.exists())

    print("Data:", DATA)
    print("Data exists:", Path(DATA).exists())

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    if not Path(DATA).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA}"
        )

    model = YOLO(str(MODEL))

    print("\nModel classes:")
    print(model.names)

    results = model.val(
        data=DATA,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        conf=0.25,
        plots=True,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("V2-A MULTIFIRE OBJECT EVALUATION COMPLETE")
    print("=" * 70)

    metrics = results.results_dict

    print(
        f"Precision   : "
        f"{metrics.get('metrics/precision(B)', 0.0):.4f}"
    )

    print(
        f"Recall      : "
        f"{metrics.get('metrics/recall(B)', 0.0):.4f}"
    )

    print(
        f"mAP50       : "
        f"{metrics.get('metrics/mAP50(B)', 0.0):.4f}"
    )

    print(
        f"mAP50-95    : "
        f"{metrics.get('metrics/mAP50-95(B)', 0.0):.4f}"
    )

    print("\nClass names:")
    print(results.names)

    print("\nSaved to:")
    print(
        f"{PROJECT}\\{NAME}"
    )


if __name__ == "__main__":
    main()