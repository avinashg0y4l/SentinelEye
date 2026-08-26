from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


RESULT_DIR = Path(
    "runs/detect/runs/detect/results/"
    "detector_v2/multifire_urban_fire_test"
)

OUTPUT_DIR = RESULT_DIR / "comparison_visuals"

PAIR_COUNT = 3
JPEG_QUALITY = 95


def get_font(size=28):
    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]

    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(
                    str(path),
                    size=size,
                )
            except OSError:
                pass

    return ImageFont.load_default()


def resize_to_height(image, height):
    if image.height == height:
        return image

    width = int(
        image.width * height / image.height
    )

    return image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )


def make_pair(
    labels_path,
    prediction_path,
    index,
):
    gt = Image.open(
        labels_path
    ).convert("RGB")

    pred = Image.open(
        prediction_path
    ).convert("RGB")

    height = min(
        gt.height,
        pred.height,
    )

    gt = resize_to_height(
        gt,
        height,
    )

    pred = resize_to_height(
        pred,
        height,
    )

    gap = 18
    header_height = 82

    total_width = (
        gt.width
        + gap
        + pred.width
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            height + header_height,
        ),
        (245, 246, 248),
    )

    draw = ImageDraw.Draw(canvas)

    title_font = get_font(25)
    small_font = get_font(16)

    draw.text(
        (15, 10),
        "GROUND TRUTH",
        fill=(22, 101, 52),
        font=title_font,
    )

    draw.text(
        (
            gt.width + gap + 15,
            10,
        ),
        "PREDICTION",
        fill=(185, 28, 28),
        font=title_font,
    )

    draw.text(
        (15, 48),
        "Manual Fire boxes",
        fill=(107, 114, 128),
        font=small_font,
    )

    draw.text(
        (
            gt.width + gap + 15,
            48,
        ),
        "V2-A model output",
        fill=(107, 114, 128),
        font=small_font,
    )

    canvas.paste(
        gt,
        (0, header_height),
    )

    canvas.paste(
        pred,
        (
            gt.width + gap,
            header_height,
        ),
    )

    separator_x = (
        gt.width
        + gap // 2
    )

    draw.line(
        (
            separator_x,
            header_height,
            separator_x,
            canvas.height,
        ),
        fill=(209, 213, 219),
        width=3,
    )

    output = (
        OUTPUT_DIR
        / f"gt_vs_pred_batch{index}.jpg"
    )

    canvas.save(
        output,
        quality=JPEG_QUALITY,
        optimize=True,
    )

    return output


def make_overview(paths):
    images = [
        Image.open(path).convert("RGB")
        for path in paths
    ]

    if not images:
        return None

    width = max(
        image.width
        for image in images
    )

    gap = 18

    total_height = (
        sum(
            image.height
            for image in images
        )
        + gap * (len(images) - 1)
    )

    overview = Image.new(
        "RGB",
        (
            width,
            total_height,
        ),
        (245, 246, 248),
    )

    y = 0

    for index, image in enumerate(
        images
    ):
        overview.paste(
            image,
            (0, y),
        )

        y += image.height

        if index < len(images) - 1:
            y += gap

    output = (
        OUTPUT_DIR
        / "gt_vs_pred_overview.jpg"
    )

    overview.save(
        output,
        quality=JPEG_QUALITY,
        optimize=True,
    )

    return output


def main():
    print("=" * 72)
    print(
        "SentinelEye - MultiFire "
        "GT vs Prediction Visualization"
    )
    print("=" * 72)

    if not RESULT_DIR.exists():
        raise FileNotFoundError(
            "Evaluation directory not found:\n"
            f"{RESULT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    created = []

    for index in range(
        PAIR_COUNT
    ):
        labels = (
            RESULT_DIR
            / f"val_batch{index}_labels.jpg"
        )

        predictions = (
            RESULT_DIR
            / f"val_batch{index}_pred.jpg"
        )

        if not labels.exists():
            print(
                f"Missing: {labels.name}"
            )
            continue

        if not predictions.exists():
            print(
                f"Missing: "
                f"{predictions.name}"
            )
            continue

        output = make_pair(
            labels,
            predictions,
            index,
        )

        created.append(output)

        print(
            f"Created: {output}"
        )

    if not created:
        raise RuntimeError(
            "No GT/prediction pairs found."
        )

    overview = make_overview(
        created
    )

    print("\n" + "=" * 72)
    print("VISUALIZATION COMPLETE")
    print("=" * 72)

    print("\nIndividual comparisons:")

    for path in created:
        print(path)

    if overview:
        print("\nOverview:")
        print(overview)


if __name__ == "__main__":
    main()