from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "runs/detect/runs/detect/results/"
    "detector_v2/dfire_multifire_manual/"
    "weights/best.pt"
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


# ============================================================
# SIMPLE THEME
# ============================================================

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

        .app-title {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.15rem;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 14px;
        }

        .card-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 650;
            margin-bottom: 10px;
        }

        .metric-label {
            color: #6b7280;
            font-size: 0.78rem;
            margin-bottom: 3px;
        }

        .metric-value {
            color: #111827;
            font-size: 1.45rem;
            font-weight: 700;
        }

        .status-safe {
            color: #15803d;
        }

        .status-smoke {
            color: #b45309;
        }

        .status-fire {
            color: #b91c1c;
        }

        .footer {
            color: #9ca3af;
            font-size: 0.78rem;
            text-align: center;
            margin-top: 1.5rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }

            .app-title {
                font-size: 1.65rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def status_for(fire_count: int, smoke_count: int):
    if fire_count:
        return "FIRE ALERT", "status-fire"
    if smoke_count:
        return "SMOKE DETECTED", "status-smoke"
    return "NO THREAT DETECTED", "status-safe"


def draw_detections(image_bgr: np.ndarray, result):
    output = image_bgr.copy()

    fire_count = 0
    smoke_count = 0
    confidences = []

    if result.boxes is None:
        return output, fire_count, smoke_count, confidences

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    height, width = output.shape[:2]

    for box, cls, score in zip(boxes, classes, scores):
        class_id = int(cls)
        confidence = float(score)

        x1, y1, x2, y2 = [int(v) for v in box]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        if class_id == 1:
            label = "Fire"
            fire_count += 1
            color = (50, 50, 220)       # BGR
        elif class_id == 0:
            label = "Smoke"
            smoke_count += 1
            color = (0, 140, 230)       # BGR
        else:
            continue

        confidences.append(confidence)

        # Clean, consistent box.
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
            cv2.LINE_AA,
        )

        text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(0.7, min(width, height) / 900))
        thickness = 1 if min(width, height) < 600 else 2

        (tw, th), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        label_top = y1 - th - baseline - 6
        if label_top < 0:
            label_top = y1

        label_bottom = min(
            height - 1,
            label_top + th + baseline + 6,
        )

        label_right = min(
            width - 1,
            x1 + tw + 10,
        )

        cv2.rectangle(
            output,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )

        text_y = label_bottom - baseline - 3

        cv2.putText(
            output,
            text,
            (x1 + 5, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output, fire_count, smoke_count, confidences


def show_metrics(
    fire_count: int,
    smoke_count: int,
    confidences,
    inference_ms: float,
):
    status, status_class = status_for(
        fire_count,
        smoke_count,
    )

    highest = max(confidences) if confidences else 0.0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">STATUS</div>
                <div class="metric-value {status_class}">
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
                <div class="metric-label">FIRE</div>
                <div class="metric-value">{fire_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">SMOKE</div>
                <div class="metric-value">{smoke_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        value = f"{highest:.2f}" if confidences else f"{inference_ms:.0f} ms"
        label = "TOP CONFIDENCE" if confidences else "INFERENCE"

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🔥 SentinelEye</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Fire and smoke detection for UAV and smart-city monitoring.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
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
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
    )

    source = st.radio(
        "Input",
        [
            "Image",
            "Video",
            "Camera",
        ],
    )

    st.divider()

    st.caption("Model")
    st.write("YOLO11n")
    st.write("Classes: Smoke, Fire")
    st.write(f"Input: {IMG_SIZE}px")


# ============================================================
# IMAGE MODE
# ============================================================

if source == "Image":
    st.markdown(
        '<div class="card-title">Image detection</div>',
        unsafe_allow_html=True,
    )

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

        start = time.perf_counter()

        result = model.predict(
            image_bgr,
            imgsz=IMG_SIZE,
            conf=confidence,
            verbose=False,
        )[0]

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        annotated_bgr, fire_count, smoke_count, confidences = (
            draw_detections(
                image_bgr,
                result,
            )
        )

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )

        show_metrics(
            fire_count,
            smoke_count,
            confidences,
            inference_ms,
        )

        st.image(
            annotated_rgb,
            use_container_width=True,
        )


# ============================================================
# CAMERA MODE
# ============================================================

elif source == "Camera":
    st.markdown(
        '<div class="card-title">Camera snapshot</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Capture a frame from the browser camera and run detection."
    )

    camera = st.camera_input("Capture frame")

    if camera:
        image = Image.open(camera).convert("RGB")
        image_rgb = np.asarray(image)

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        start = time.perf_counter()

        result = model.predict(
            image_bgr,
            imgsz=IMG_SIZE,
            conf=confidence,
            verbose=False,
        )[0]

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        annotated_bgr, fire_count, smoke_count, confidences = (
            draw_detections(
                image_bgr,
                result,
            )
        )

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )

        show_metrics(
            fire_count,
            smoke_count,
            confidences,
            inference_ms,
        )

        st.image(
            annotated_rgb,
            use_container_width=True,
        )


# ============================================================
# VIDEO MODE
# ============================================================

elif source == "Video":
    st.markdown(
        '<div class="card-title">Video detection</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a video and SentinelEye will process it frame by frame."
    )

    uploaded = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as temp_file:
            temp_file.write(uploaded.read())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            st.error("Could not open the selected video.")
            st.stop()

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        progress = st.progress(0.0)

        frame_number = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            start = time.perf_counter()

            result = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=confidence,
                verbose=False,
            )[0]

            inference_ms = (
                time.perf_counter() - start
            ) * 1000

            annotated_bgr, fire_count, smoke_count, confidences = (
                draw_detections(
                    frame,
                    result,
                )
            )

            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB,
            )

            frame_placeholder.image(
                annotated_rgb,
                use_container_width=True,
            )

            status, _ = status_for(
                fire_count,
                smoke_count,
            )

            status_placeholder.write(
                f"**{status}**  ·  "
                f"Fire: **{fire_count}**  ·  "
                f"Smoke: **{smoke_count}**  ·  "
                f"Inference: **{inference_ms:.0f} ms**"
            )

            if total_frames > 0:
                progress.progress(
                    min(
                        1.0,
                        frame_number / total_frames,
                    )
                )

        cap.release()

        st.success("Video analysis complete.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'SentinelEye · YOLO11n Fire & Smoke Detection'
    '</div>',
    unsafe_allow_html=True,
)
