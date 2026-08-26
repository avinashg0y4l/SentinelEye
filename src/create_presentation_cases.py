from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESULT_DIR = Path(
    "runs/detect/runs/detect/results/"
    "detector_v2/multifire_urban_fire_test"
)

OUTPUT_DIR = (
    RESULT_DIR
    / "presentation_cases"
)


# ------------------------------------------------------------
# Selected tiles from the saved 3x3 evaluation montages.
#
# Tile numbering:
#   1  2  3
#   4  5  6
#   7  8  9
#
# Case 1: strong successful detection
# Case 2: detection with localization / duplicate issue
# Case 3: smoke-heavy edge case
# ------------------------------------------------------------

CASES = [
    {
        "name": "case_01_success",
        "batch": 0,
        "tile": 2,
        "title": "Case 1 - Successful Fire Detection",
        "description": (
            "The model correctly identifies the main fire region "
            "with strong confidence."
        ),
    },
    {
        "name": "case_02_localization",
        "batch": 1,
        "tile": 2,
        "title": "Case 2 - Detection with Localization Error",
        "description": (
            "The fire event is detected, but the prediction includes "
            "multiple/overlapping boxes and localization is imperfect."
        ),
    },
    {
        "name": "case_03_smoke_edge",
        "batch": 2,
        "tile": 1,
        "title": "Case 3 - Smoke-Heavy Edge Case",
        "description": (
            "A smoke-dominant UAV scene. The model detects the fire event, "
            "but the localization is broad. MultiFire evaluation labels "
            "this test subset as Fire-only."
        ),
    },
]


def get_font(size):
    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]

    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(
                    str(font_path),
                    size=size,
                )
            except OSError:
                pass

    return ImageFont.load_default()


def crop_tile(image, tile_number):
    """
    Crop one tile from a 3x3 YOLO validation montage.
    """

    width, height = image.size

    columns = 3
    rows = 3

    col = (tile_number - 1) % columns
    row = (tile_number - 1) // columns

    x1 = int(width * col / columns)
    x2 = int(width * (col + 1) / columns)

    y1 = int(height * row / rows)
    y2 = int(height * (row + 1) / rows)

    return image.crop(
        (x1, y1, x2, y2)
    )


def make_case(case):
    batch = case["batch"]
    tile = case["tile"]

    labels_path = (
        RESULT_DIR
        / f"val_batch{batch}_labels.jpg"
    )

    pred_path = (
        RESULT_DIR
        / f"val_batch{batch}_pred.jpg"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing labels image:\n{labels_path}"
        )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing prediction image:\n{pred_path}"
        )

    labels = Image.open(
        labels_path
    ).convert("RGB")

    pred = Image.open(
        pred_path
    ).convert("RGB")

    labels_tile = crop_tile(
        labels,
        tile,
    )

    pred_tile = crop_tile(
        pred,
        tile,
    )

    # Put both panels at the same size.
    height = min(
        labels_tile.height,
        pred_tile.height,
    )

    def resize_height(image, target_height):
        if image.height == target_height:
            return image

        new_width = int(
            image.width
            * target_height
            / image.height
        )

        return image.resize(
            (new_width, target_height),
            Image.Resampling.LANCZOS,
        )

    labels_tile = resize_height(
        labels_tile,
        height,
    )

    pred_tile = resize_height(
        pred_tile,
        height,
    )

    gap = 16

    title_height = 115

    panel_width = (
        labels_tile.width
        + gap
        + pred_tile.width
    )

    canvas = Image.new(
        "RGB",
        (
            panel_width,
            height + title_height,
        ),
        (248, 249, 250),
    )

    draw = ImageDraw.Draw(canvas)

    title_font = get_font(30)
    subtitle_font = get_font(18)
    panel_font = get_font(22)

    # Title.
    draw.text(
        (20, 12),
        case["title"],
        fill=(17, 24, 39),
        font=title_font,
    )

    # Description.
    draw.text(
        (20, 52),
        case["description"],
        fill=(75, 85, 99),
        font=subtitle_font,
    )

    # Panel labels.
    left_x = 10

    right_x = (
        labels_tile.width
        + gap
        + 10
    )

    draw.text(
        (left_x, 86),
        "GROUND TRUTH",
        fill=(22, 101, 52),
        font=panel_font,
    )

    draw.text(
        (right_x, 86),
        "V2-A PREDICTION",
        fill=(185, 28, 28),
        font=panel_font,
    )

    # Paste images.
    image_y = title_height

    canvas.paste(
        labels_tile,
        (0, image_y),
    )

    canvas.paste(
        pred_tile,
        (
            labels_tile.width + gap,
            image_y,
        ),
    )

    # Separator.
    separator_x = (
        labels_tile.width
        + gap // 2
    )

    draw.line(
        (
            separator_x,
            image_y,
            separator_x,
            canvas.height,
        ),
        fill=(209, 213, 219),
        width=3,
    )

    output = (
        OUTPUT_DIR
        / f"{case['name']}.jpg"
    )

    canvas.save(
        output,
        quality=95,
        optimize=True,
    )

    return output


def make_overview(case_paths):
    images = [
        Image.open(
            path
        ).convert("RGB")
        for path in case_paths
    ]

    if not images:
        return None

    width = max(
        image.width
        for image in images
    )

    gap = 20

    total_height = (
        sum(
            image.height
            for image in images
        )
        + gap
        * (len(images) - 1)
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
        / "presentation_cases_overview.jpg"
    )

    overview.save(
        output,
        quality=95,
        optimize=True,
    )

    return output


def main():
    print("=" * 72)
    print(
        "SentinelEye - Presentation Evidence Cases"
    )
    print("=" * 72)

    if not RESULT_DIR.exists():
        raise FileNotFoundError(
            "MultiFire evaluation directory not found:\n"
            f"{RESULT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    created = []

    for case in CASES:

        print(
            f"\nCreating: "
            f"{case['title']}"
        )

        output = make_case(
            case
        )

        created.append(output)

        print(
            f"Saved: {output}"
        )

    overview = make_overview(
        created
    )

    print("\n" + "=" * 72)
    print("PRESENTATION CASES COMPLETE")
    print("=" * 72)

    print("\nCreated files:")

    for path in created:
        print(path)

    if overview:
        print("\nOverview:")
        print(overview)


if __name__ == "__main__":
    main()