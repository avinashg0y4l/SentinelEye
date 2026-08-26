from ultralytics import YOLO

DATA = "data/experiments/detector_v1/DFIRE_clean_split/data.yaml"

model = YOLO("yolo11n.pt")

model.train(
    data=DATA,
    epochs=1,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    project="results/detector_v1",
    name="smoke_test",
    exist_ok=True,
)