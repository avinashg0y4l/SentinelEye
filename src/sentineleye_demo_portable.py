from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PORTABLE PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "sentineleye_v2a.pt"
)

IMG_SIZE = 640
DEFAULT_CONFIDENCE = 0.40


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SentinelEye",
    page_icon="🔥",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f7f8;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .title {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 4px;
        }

        .subtitle {
            color: #6b7280;
            margin-bottom: 24px;
        }

        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
        }

        .label {
            color: #6b7280;
            font-size: 12px;
            text-transform: uppercase;
        }

        .value {
            color: #111827;
            font-size: 22px;
            font-weight: 700;
        }

        .fire {
            color: #b91c1c;
        }

        .smoke {
            color: #b45309;
        }

        .safe {
            color: #15803d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🔥 SentinelEye</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Fire and smoke detection for UAV and smart-city monitoring.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Model file not found.\n\n"
        f"Expected:\n{MODEL_PATH}"
    )
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### SentinelEye")

    confidence = st.slider(
        "Confidence threshold",
        0.10,
        0.90,
        DEFAULT_CONFIDENCE,
        0.05,
    )

    input_mode = st.radio(
        "Input",
        [
            "Image",
            "Camera",
        ],
    )

    st.divider()

    st.caption("Model")
    st.write("YOLO11n")
    st.write("Smoke / Fire")
    st.write("640 px")


# ============================================================
# DETECTION
# ============================================================

def detect(image_bgr):
    start = time.perf_counter()

    result = model.predict(
        image_bgr,
        imgsz=IMG_SIZE,
        conf=confidence,
        verbose=False,
    )[0]

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    output = image_bgr.copy()

    fire_count = 0
    smoke_count = 0
    confidences = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        height, width = output.shape[:2]

        for box, cls, score in zip(
            boxes,
            classes,
            scores,
        ):
            class_id = int(cls)
            score = float(score)

            x1, y1, x2, y2 = [
                int(v) for v in box
            ]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            if class_id == 1:
                label = "Fire"
                fire_count += 1
                color = (40, 60, 220)
            elif class_id == 0:
                label = "Smoke"
                smoke_count += 1
                color = (0, 140, 230)
            else:
                continue

            confidences.append(score)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            text = f"{label} {score:.2f}"

            cv2.putText(
                output,
                text,
                (x1 + 5, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    return (
        output,
        fire_count,
        smoke_count,
        confidences,
        elapsed_ms,
    )


def show_metrics(
    fire_count,
    smoke_count,
    confidences,
    elapsed_ms,
):
    if fire_count:
        status = "FIRE ALERT"
        status_class = "fire"
    elif smoke_count:
        status = "SMOKE DETECTED"
        status_class = "smoke"
    else:
        status = "NO THREAT DETECTED"
        status_class = "safe"

    top_conf = (
        max(confidences)
        if confidences
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Status</div>
                <div class="value {status_class}">
                    {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Fire</div>
                <div class="value">{fire_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Smoke</div>
                <div class="value">{smoke_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        value = (
            f"{top_conf:.2f}"
            if confidences
            else f"{elapsed_ms:.0f} ms"
        )

        label = (
            "Top confidence"
            if confidences
            else "Inference"
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# IMAGE MODE
# ============================================================

if input_mode == "Image":

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")
        image_rgb = np.asarray(image)

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        (
            annotated,
            fire_count,
            smoke_count,
            confidences,
            elapsed_ms,
        ) = detect(image_bgr)

        show_metrics(
            fire_count,
            smoke_count,
            confidences,
            elapsed_ms,
        )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

        st.image(
            annotated_rgb,
            use_container_width=True,
        )


# ============================================================
# CAMERA MODE
# ============================================================

else:

    st.caption(
        "Capture one frame using your browser camera."
    )

    camera = st.camera_input(
        "Capture frame"
    )

    if camera:

        image = Image.open(camera).convert("RGB")
        image_rgb = np.asarray(image)

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        (
            annotated,
            fire_count,
            smoke_count,
            confidences,
            elapsed_ms,
        ) = detect(image_bgr)

        show_metrics(
            fire_count,
            smoke_count,
            confidences,
            elapsed_ms,
        )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

        st.image(
            annotated_rgb,
            use_container_width=True,
        )