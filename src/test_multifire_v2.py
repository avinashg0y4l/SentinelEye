from pathlib import Path
import json

from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v2/dfire_multifire_manual/"
    "weights/best.pt"
)

MULTIFIRE_ROOT = Path(
    "data/processed/Multifire/Test/Test"
)

CONF_THRESHOLD = 0.25


def evaluate_folder(model, folder_name):
    folder = MULTIFIRE_ROOT / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    images = sorted(folder.glob("*.jpg"))

    fire_images = 0
    smoke_images = 0
    any_images = 0

    total_fire_predictions = 0
    total_smoke_predictions = 0

    print(f"\nTesting {folder_name}")
    print("Images:", len(images))

    for index, image_path in enumerate(images, start=1):

        results = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONF_THRESHOLD,
            device=0,
            verbose=False,
        )

        result = results[0]

        has_fire = False
        has_smoke = False

        if result.boxes is not None:
            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            for cls in classes:
                cls_id = int(cls)

                if cls_id == 1:
                    has_fire = True
                    total_fire_predictions += 1

                elif cls_id == 0:
                    has_smoke = True
                    total_smoke_predictions += 1

        if has_fire:
            fire_images += 1

        if has_smoke:
            smoke_images += 1

        if has_fire or has_smoke:
            any_images += 1

        if index % 100 == 0:
            print(
                f"Processed: "
                f"{index}/{len(images)}"
            )

    count = len(images)

    result = {
        "images": count,
        "images_with_fire_detection": fire_images,
        "images_with_smoke_detection": smoke_images,
        "images_with_any_detection": any_images,
        "fire_detection_rate": (
            fire_images / count
            if count else 0.0
        ),
        "smoke_detection_rate": (
            smoke_images / count
            if count else 0.0
        ),
        "any_detection_rate": (
            any_images / count
            if count else 0.0
        ),
        "total_fire_predictions":
            total_fire_predictions,
        "total_smoke_predictions":
            total_smoke_predictions,
    }

    return result


def main():

    print("=" * 70)
    print("SentinelEye - V2-A MultiFire Cross-Domain Evaluation")
    print("=" * 70)

    print("Model:")
    print(MODEL)
    print("Model exists:", MODEL.exists())

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    model = YOLO(str(MODEL))

    print("\nModel classes:")
    print(model.names)

    fu = evaluate_folder(
        model,
        "FU"
    )

    nu = evaluate_folder(
        model,
        "NU"
    )

    report = {
        "model": str(MODEL),
        "confidence_threshold":
            CONF_THRESHOLD,
        "FU_Urban_Fire": fu,
        "NU_Urban_Normal": nu,
    }

    output_dir = Path(
        "results/detector_v2/"
        "multifire_urban"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("V2-A MULTIFIRE EVALUATION COMPLETE")
    print("=" * 70)

    print("\nFU - Urban Fire")
    print(
        f"Images                  : "
        f"{fu['images']}"
    )
    print(
        f"Fire detection rate     : "
        f"{fu['fire_detection_rate'] * 100:.2f}%"
    )
    print(
        f"Smoke detection rate    : "
        f"{fu['smoke_detection_rate'] * 100:.2f}%"
    )
    print(
        f"Any detection rate      : "
        f"{fu['any_detection_rate'] * 100:.2f}%"
    )
    print(
        f"Total Fire predictions  : "
        f"{fu['total_fire_predictions']}"
    )
    print(
        f"Total Smoke predictions : "
        f"{fu['total_smoke_predictions']}"
    )

    print("\nNU - Urban Normal")
    print(
        f"Images                  : "
        f"{nu['images']}"
    )
    print(
        f"Fire false-alarm rate   : "
        f"{nu['fire_detection_rate'] * 100:.2f}%"
    )
    print(
        f"Smoke false-alarm rate  : "
        f"{nu['smoke_detection_rate'] * 100:.2f}%"
    )
    print(
        f"Any detection rate      : "
        f"{nu['any_detection_rate'] * 100:.2f}%"
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()