from ultralytics import YOLO


MODEL = "yolo11n.pt"

DATA = (
    "data/experiments/detector_v2/"
    "combined_v2_clean/data.yaml"
)

PROJECT = (
    "runs/detect/results/detector_v2"
)

NAME = "dfire_multifire_manual"


def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=20,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=8,
        pretrained=True,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()