from pathlib import Path
import json

from ultralytics import YOLO


MODEL = Path(
    "runs/detect/runs/detect/results/"
    "detector_v1-2/weights/best.pt"
)

ROOT = Path(
    "data/processed/Multifire/Test/Test"
)
CLASS_NAMES = {
    0: "Smoke",
    1: "Fire",
}

CONF_THRESHOLD = 0.25


def get_images(folder):
    extensions = {".jpg", ".jpeg", ".png"}
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]


def evaluate_folder(model, folder_name):
    folder = ROOT / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    images = get_images(folder)

    total_images = len(images)
    images_with_fire = 0
    images_with_smoke = 0
    images_with_any_detection = 0

    total_fire_predictions = 0
    total_smoke_predictions = 0

    for index, image_path in enumerate(
        images,
        start=1
    ):

        results = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONF_THRESHOLD,
            device=0,
            verbose=False,
        )

        result = results[0]

        detected_fire = False
        detected_smoke = False

        if result.boxes is not None:

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            for cls in classes:

                class_id = int(cls)

                if class_id == 1:
                    detected_fire = True
                    total_fire_predictions += 1

                elif class_id == 0:
                    detected_smoke = True
                    total_smoke_predictions += 1

        if detected_fire:
            images_with_fire += 1

        if detected_smoke:
            images_with_smoke += 1

        if detected_fire or detected_smoke:
            images_with_any_detection += 1

        if index % 500 == 0:
            print(
                f"{folder_name}: "
                f"{index}/{total_images}"
            )

    fire_rate = (
        images_with_fire / total_images
        if total_images
        else 0
    )

    smoke_rate = (
        images_with_smoke / total_images
        if total_images
        else 0
    )

    detection_rate = (
        images_with_any_detection / total_images
        if total_images
        else 0
    )

    return {
        "images": total_images,
        "images_with_fire_detection": images_with_fire,
        "images_with_smoke_detection": images_with_smoke,
        "images_with_any_detection": images_with_any_detection,
        "fire_detection_rate": fire_rate,
        "smoke_detection_rate": smoke_rate,
        "any_detection_rate": detection_rate,
        "total_fire_predictions": total_fire_predictions,
        "total_smoke_predictions": total_smoke_predictions,
    }


def main():

    print("=" * 70)
    print("SentinelEye - MultiFire20K Urban Cross-Domain Test")
    print("=" * 70)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}"
        )

    model = YOLO(str(MODEL))

    print("\nModel:")
    print(MODEL)

    print("\nTesting FU = Urban Fire")
    fu_results = evaluate_folder(
        model,
        "FU"
    )

    print("\nTesting NU = Urban Normal")
    nu_results = evaluate_folder(
        model,
        "NU"
    )

    report = {
        "model": str(MODEL),
        "confidence_threshold": CONF_THRESHOLD,
        "FU_Urban_Fire": fu_results,
        "NU_Urban_Normal": nu_results,
    }

    output_dir = Path(
        "results/detector_v1/multifire_urban"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        output_dir / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("CROSS-DOMAIN TEST COMPLETE")
    print("=" * 70)

    print("\nFU - Urban Fire")
    print(
        f"Images: "
        f"{fu_results['images']}"
    )
    print(
        f"Images with Fire detection: "
        f"{fu_results['images_with_fire_detection']}"
    )
    print(
        f"Fire detection rate: "
        f"{fu_results['fire_detection_rate']:.3f}"
    )

    print("\nNU - Urban Normal")
    print(
        f"Images: "
        f"{nu_results['images']}"
    )
    print(
        f"Images with Fire detection: "
        f"{nu_results['images_with_fire_detection']}"
    )
    print(
        f"False Fire rate: "
        f"{nu_results['fire_detection_rate']:.3f}"
    )

    print("\nReport:")
    print(report_path)


if __name__ == "__main__":
    main()