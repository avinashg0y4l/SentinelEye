from ultralytics import YOLO


DATA = (
    "data/experiments/detector_v1/"
    "DFIRE_clean_split/data.yaml"
)

model = YOLO("yolo11n.pt")

model.train(
    data=DATA,
    epochs=20,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    amp=True,
    patience=8,
    project="runs/detect/results",
    name="detector_v1",
    exist_ok=False,
)