from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR.parent

MODEL_PATH = (
    DEMO_DIR
    / "models"
    / "sentineleye_v2a.pt"
)

EVIDENCE_DIR = (
    DEMO_DIR
    / "evidence"
)

IMG_SIZE = 640
DEFAULT_CONF = 0.40


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SentinelEye Operations",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD STYLESHEET
# ============================================================

CSS_PATH = APP_DIR / "style.css"

if not CSS_PATH.exists():
    st.error(
        f"Stylesheet not found:\n{CSS_PATH}"
    )
    st.stop()

with open(
    CSS_PATH,
    "r",
    encoding="utf-8",
) as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True,
    )

# ============================================================
# SESSION STATE
# ============================================================

if "incidents" not in st.session_state:
    st.session_state.incidents = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "SentinelEye model was not found.\n\n"
        f"Expected:\n{MODEL_PATH}"
    )
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# HELPERS
# ============================================================

def model_device():
    try:
        import torch

        if torch.cuda.is_available():
            return 0

    except Exception:
        pass

    return "cpu"


def run_detection(image_bgr, confidence):
    start = time.perf_counter()

    result = model.predict(
        source=image_bgr,
        imgsz=IMG_SIZE,
        conf=confidence,
        device=model_device(),
        verbose=False,
    )[0]

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return result, elapsed_ms


def annotate_image(image_bgr, result):
    output = image_bgr.copy()

    detections = []
    fire_count = 0
    smoke_count = 0

    if result.boxes is None:
        return (
            output,
            detections,
            fire_count,
            smoke_count,
        )

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    height, width = output.shape[:2]

    for box, cls, conf in zip(
        boxes,
        classes,
        confidences,
    ):

        class_id = int(cls)
        score = float(conf)

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
            name = "Fire"
            color = (40, 60, 210)
            fire_count += 1

        elif class_id == 0:
            name = "Smoke"
            color = (0, 130, 220)
            smoke_count += 1

        else:
            continue

        detections.append(
            {
                "class": name,
                "confidence": score,
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
            cv2.LINE_AA,
        )

        label = f"{name} {score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2

        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_y1 = max(
            0,
            y1 - text_height - baseline - 8,
        )

        label_y2 = (
            label_y1
            + text_height
            + baseline
            + 8
        )

        label_x2 = min(
            width - 1,
            x1 + text_width + 10,
        )

        cv2.rectangle(
            output,
            (x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 5,
                label_y2 - baseline - 3,
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return (
        output,
        detections,
        fire_count,
        smoke_count,
    )


def get_status(fire_count, smoke_count):
    if fire_count > 0:
        return "FIRE ALERT", "fire"

    if smoke_count > 0:
        return "SMOKE DETECTED", "smoke"

    return "NO THREAT", "safe"


def add_incident(
    source,
    fire_count,
    smoke_count,
    detections,
    inference_ms,
):
    if fire_count == 0 and smoke_count == 0:
        return

    status, _ = get_status(
        fire_count,
        smoke_count,
    )

    max_conf = (
        max(
            d["confidence"]
            for d in detections
        )
        if detections
        else 0.0
    )

    event = {
        "time": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "source": source,
        "status": status,
        "fire": fire_count,
        "smoke": smoke_count,
        "confidence": round(
            max_conf,
            3,
        ),
        "inference_ms": round(
            inference_ms,
            1,
        ),
    }

    st.session_state.incidents.insert(
        0,
        event,
    )

    st.session_state.incidents = (
        st.session_state.incidents[:50]
    )


def display_kpi(
    label,
    value,
    css_class="",
):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {css_class}">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================
# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:
    st.markdown(
        """
        <div class="page-header">
            <div class="page-header-title">
                SentinelEye Operations
            </div>
            <div class="page-header-subtitle">
                UAV-based fire and smoke monitoring
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div class="status-box">
            ● SYSTEM ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

st.markdown(
    """
    <div class="nav-title">SentinelEye</div>
    <div class="nav-label">OPERATIONS</div>
    """,
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(
    [1, 1.15, 0.9, 1.45, 0.95, 0.8]
)

def nav_button(label, key, value):
    if st.button(
        label,
        key=key,
        width="stretch",
        type="primary" if st.session_state.page == value else "secondary",
    ):
        st.session_state.page = value
        st.rerun()

with nav1:
    nav_button("Overview", "nav_overview", "Overview")

with nav2:
    nav_button("Live Detection", "nav_live", "Live Detection")

with nav3:
    nav_button("Alerts", "nav_alerts", "Alerts")

with nav4:
    nav_button("Model & Evaluation", "nav_model", "Model & Evaluation")

with nav5:
    nav_button("Evidence", "nav_evidence", "Evidence")

with nav6:
    nav_button("System", "nav_system", "System")

page = st.session_state.page

settings_left, settings_right = st.columns([2.5, 1])

with settings_left:
    st.markdown(
        """
        <div class="settings-shell">
            <div class="settings-label">DETECTION CONFIDENCE</div>
        """,
        unsafe_allow_html=True,
    )

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONF,
        step=0.05,
        label_visibility="collapsed",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with settings_right:
    st.markdown(
        """
        <div class="settings-shell">
            <div class="settings-label">ACTIVE MODEL</div>
            <div class="sidebar-info">
                <b>YOLO11n V2-A</b><br>
                Fire / Smoke · 640 px
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        'Operations Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    total_alerts = len(
        st.session_state.incidents
    )

    fire_alerts = sum(
        1
        for x in st.session_state.incidents
        if x["status"] == "FIRE ALERT"
    )

    smoke_alerts = sum(
        1
        for x in st.session_state.incidents
        if x["status"] == "SMOKE DETECTED"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        display_kpi(
            "Session alerts",
            total_alerts,
        )

    with c2:
        display_kpi(
            "Fire alerts",
            fire_alerts,
            "kpi-danger",
        )

    with c3:
        display_kpi(
            "Smoke alerts",
            smoke_alerts,
            "kpi-warning",
        )

    with c4:
        display_kpi(
            "Model",
            "YOLO11n",
            "kpi-success",
        )

    st.markdown(
        '<div class="section-title">'
        'Current Monitoring'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [2.2, 1]
    )

    with left:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    Live Detection
                </div>
                <div class="muted">
                    Upload an image or use the
                    camera from the Live Detection
                    section.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            st.session_state.last_result
            is not None
        ):

            result_data = (
                st.session_state.last_result
            )

            st.image(
                result_data["image"],
                width="stretch",
            )

        else:

            st.info(
                "No detection has been run "
                "in this session."
            )

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    Latest Status
                </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            st.session_state.last_result
            is not None
        ):

            latest = (
                st.session_state.last_result
            )

            status, status_class = get_status(
                latest["fire"],
                latest["smoke"],
            )

            if status_class == "fire":

                st.markdown(
                    f"""
                    <div class="alert-fire">
                        <b>{status}</b><br>
                        Fire detections:
                        {latest["fire"]}<br>
                        Smoke detections:
                        {latest["smoke"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif status_class == "smoke":

                st.markdown(
                    f"""
                    <div class="alert-smoke">
                        <b>{status}</b><br>
                        Smoke detections:
                        {latest["smoke"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="alert-safe">
                        <b>NO THREAT</b><br>
                        No Fire or Smoke detected.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(
                f"Confidence: "
                f"{latest['confidence']:.2f}"
            )

            st.write(
                f"Inference: "
                f"{latest['inference_ms']:.0f} ms"
            )

            st.write(
                f"Source: "
                f"{latest['source']}"
            )

        else:

            st.write(
                "Waiting for first detection."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">'
        'Recent Incidents'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.incidents:

        rows = []

        for item in (
            st.session_state.incidents[:10]
        ):

            rows.append(
                {
                    "Time": item["time"],
                    "Source": item["source"],
                    "Status": item["status"],
                    "Fire": item["fire"],
                    "Smoke": item["smoke"],
                    "Confidence": item["confidence"],
                    "Inference (ms)": item[
                        "inference_ms"
                    ],
                }
            )

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No incidents recorded in this session."
        )


# ============================================================
# LIVE DETECTION
# ============================================================

elif page == "Live Detection":

    st.markdown(
        '<div class="section-title">'
        'Live Detection'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-note">'
        'Run the V2-A model on an image, '
        'camera frame, or video.'
        '</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input source",
        [
            "Image",
            "Camera",
            "Video",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if mode == "Image":

        uploaded = st.file_uploader(
            "Select image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )

        if uploaded:

            image = Image.open(
                uploaded
            ).convert("RGB")

            image_rgb = np.asarray(
                image
            )

            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            )

            result, inference_ms = (
                run_detection(
                    image_bgr,
                    confidence,
                )
            )

            (
                annotated,
                detections,
                fire_count,
                smoke_count,
            ) = annotate_image(
                image_bgr,
                result,
            )

            max_conf = (
                max(
                    d["confidence"]
                    for d in detections
                )
                if detections
                else 0.0
            )

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            )

            st.session_state.last_result = {
                "image": annotated_rgb,
                "fire": fire_count,
                "smoke": smoke_count,
                "confidence": max_conf,
                "inference_ms": inference_ms,
                "source": uploaded.name,
            }

            add_incident(
                uploaded.name,
                fire_count,
                smoke_count,
                detections,
                inference_ms,
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                display_kpi(
                    "Status",
                    get_status(
                        fire_count,
                        smoke_count,
                    )[0],
                )

            with c2:
                display_kpi(
                    "Fire",
                    fire_count,
                    "kpi-danger"
                    if fire_count
                    else "",
                )

            with c3:
                display_kpi(
                    "Smoke",
                    smoke_count,
                    "kpi-warning"
                    if smoke_count
                    else "",
                )

            with c4:
                display_kpi(
                    "Inference",
                    f"{inference_ms:.0f} ms",
                )

            st.image(
                annotated_rgb,
                width="stretch",
            )

            if detections:

                table = []

                for index, item in enumerate(
                    detections,
                    start=1,
                ):

                    x1, y1, x2, y2 = (
                        item["box"]
                    )

                    table.append(
                        {
                            "#": index,
                            "Class": item["class"],
                            "Confidence": round(
                                item["confidence"],
                                3,
                            ),
                            "Box": (
                                f"{x1}, {y1}, "
                                f"{x2}, {y2}"
                            ),
                        }
                    )

                st.dataframe(
                    table,
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.info(
                    "No Fire or Smoke detected "
                    "above the selected confidence."
                )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    elif mode == "Camera":

        camera = st.camera_input(
            "Capture current frame"
        )

        if camera:

            image = Image.open(
                camera
            ).convert("RGB")

            image_rgb = np.asarray(
                image
            )

            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            )

            result, inference_ms = (
                run_detection(
                    image_bgr,
                    confidence,
                )
            )

            (
                annotated,
                detections,
                fire_count,
                smoke_count,
            ) = annotate_image(
                image_bgr,
                result,
            )

            max_conf = (
                max(
                    d["confidence"]
                    for d in detections
                )
                if detections
                else 0.0
            )

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            )

            st.session_state.last_result = {
                "image": annotated_rgb,
                "fire": fire_count,
                "smoke": smoke_count,
                "confidence": max_conf,
                "inference_ms": inference_ms,
                "source": "Browser camera",
            }

            add_incident(
                "Browser camera",
                fire_count,
                smoke_count,
                detections,
                inference_ms,
            )

            show1, show2 = st.columns(
                [3, 1]
            )

            with show1:

                st.image(
                    annotated_rgb,
                    width="stretch",
                )

            with show2:

                status, _ = get_status(
                    fire_count,
                    smoke_count,
                )

                st.metric(
                    "Status",
                    status,
                )

                st.metric(
                    "Fire",
                    fire_count,
                )

                st.metric(
                    "Smoke",
                    smoke_count,
                )

                st.metric(
                    "Inference",
                    f"{inference_ms:.0f} ms",
                )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    else:

        uploaded = st.file_uploader(
            "Select video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
            key="video_upload",
        )

        if uploaded:

            max_frames = st.slider(
                "Maximum frames to process",
                min_value=30,
                max_value=600,
                value=180,
                step=30,
                key="video_max_frames",
            )

            incident_cooldown = 30

            suffix = Path(uploaded.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp:
                temp.write(uploaded.getbuffer())
                input_video_path = temp.name

            cap = cv2.VideoCapture(input_video_path)

            if not cap.isOpened():
                st.error("Unable to open this video.")
            else:

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                frame_height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                if frame_width <= 0 or frame_height <= 0:
                    frame_width = 1280
                    frame_height = 720

                # Create an annotated output video.
                output_file = tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False,
                )
                output_video_path = output_file.name
                output_file.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (frame_width, frame_height),
                )

                if not writer.isOpened():
                    cap.release()
                    st.error(
                        "Unable to create the processed video."
                    )
                    st.stop()

                preview_slot = st.empty()
                status_slot = st.empty()
                progress_slot = st.progress(0.0)

                processed = 0
                last_incident_frame = -incident_cooldown

                best_confidence = 0.0
                best_frame_image = None
                best_fire_count = 0
                best_smoke_count = 0
                best_inference_ms = 0.0
                best_frame_number = None

                denominator = min(
                    total_frames,
                    max_frames,
                )

                while processed < max_frames:

                    ok, frame = cap.read()

                    if not ok:
                        break

                    result, inference_ms = run_detection(
                        frame,
                        confidence,
                    )

                    (
                        annotated,
                        detections,
                        fire_count,
                        smoke_count,
                    ) = annotate_image(
                        frame,
                        result,
                    )

                    writer.write(annotated)

                    annotated_rgb = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB,
                    )

                    current_confidence = (
                        max(
                            d["confidence"]
                            for d in detections
                        )
                        if detections
                        else 0.0
                    )

                    # Local preview. On hosted Streamlit this may
                    # update less frequently than local execution,
                    # but the complete annotated video is preserved.
                    preview_slot.image(
                        annotated_rgb,
                        width="stretch",
                    )

                    status, _ = get_status(
                        fire_count,
                        smoke_count,
                    )

                    video_time = (
                        processed / fps
                        if fps > 0
                        else 0.0
                    )

                    status_slot.markdown(
                        f"""
**{status}**

Frame: **{processed + 1} / {denominator}**  
Video time: **{video_time:.1f}s**  
Fire: **{fire_count}** · Smoke: **{smoke_count}**  
Confidence: **{current_confidence:.2f}** ·
Inference: **{inference_ms:.0f} ms**
"""
                    )

                    # Keep strongest detection for Overview.
                    if current_confidence > best_confidence:
                        best_confidence = current_confidence
                        best_frame_image = annotated_rgb.copy()
                        best_fire_count = fire_count
                        best_smoke_count = smoke_count
                        best_inference_ms = inference_ms
                        best_frame_number = processed + 1

                    # Log an incident, but not every frame.
                    if (
                        fire_count > 0
                        or smoke_count > 0
                    ) and (
                        processed - last_incident_frame
                        >= incident_cooldown
                    ):

                        add_incident(
                            f"{uploaded.name} "
                            f"[frame {processed + 1}]",
                            fire_count,
                            smoke_count,
                            detections,
                            inference_ms,
                        )

                        last_incident_frame = processed

                    processed += 1

                    if denominator > 0:
                        progress_slot.progress(
                            min(
                                1.0,
                                processed / denominator,
                            )
                        )

                cap.release()
                writer.release()

                # Save the strongest processed frame into
                # Overview -> Current Monitoring.
                if best_frame_image is not None:
                    st.session_state.last_result = {
                        "image": best_frame_image,
                        "fire": best_fire_count,
                        "smoke": best_smoke_count,
                        "confidence": best_confidence,
                        "inference_ms": best_inference_ms,
                        "source": (
                            f"{uploaded.name} "
                            f"· best frame {best_frame_number}"
                        ),
                    }
                else:
                    # Even when there are no detections, keep a
                    # valid latest frame for the Overview page.
                    try:
                        cap_preview = cv2.VideoCapture(
                            output_video_path
                        )
                        ok_preview, preview_frame = (
                            cap_preview.read()
                        )
                        cap_preview.release()

                        if ok_preview:
                            preview_rgb = cv2.cvtColor(
                                preview_frame,
                                cv2.COLOR_BGR2RGB,
                            )

                            st.session_state.last_result = {
                                "image": preview_rgb,
                                "fire": 0,
                                "smoke": 0,
                                "confidence": 0.0,
                                "inference_ms": 0.0,
                                "source": uploaded.name,
                            }
                    except Exception:
                        pass

                st.markdown(
                    '<div class="section-title">'
                    'Processed Video'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.video(
                    output_video_path,
                )

                st.success(
                    f"Processed {processed} frames "
                    f"from {uploaded.name}."
                )

# ============================================================
# ALERTS
# ============================================================

elif page == "Alerts":

    st.markdown(
        '<div class="section-title">'
        'Alerts & Incident Log'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.incidents:

        st.success(
            "No alerts have been generated "
            "in this session."
        )

    else:

        fire_count = sum(
            1
            for x in st.session_state.incidents
            if x["status"] == "FIRE ALERT"
        )

        smoke_count = sum(
            1
            for x in st.session_state.incidents
            if x["status"] == "SMOKE DETECTED"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            display_kpi(
                "Total incidents",
                len(
                    st.session_state.incidents
                ),
            )

        with c2:

            display_kpi(
                "Fire alerts",
                fire_count,
                "kpi-danger",
            )

        with c3:

            display_kpi(
                "Smoke alerts",
                smoke_count,
                "kpi-warning",
            )

        rows = []

        for item in (
            st.session_state.incidents
        ):

            rows.append(
                {
                    "Time": item["time"],
                    "Source": item["source"],
                    "Status": item["status"],
                    "Fire": item["fire"],
                    "Smoke": item["smoke"],
                    "Confidence": item["confidence"],
                    "Inference (ms)": item[
                        "inference_ms"
                    ],
                }
            )

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

        if st.button(
            "Clear session incident log"
        ):

            st.session_state.incidents = []

            st.rerun()


# ============================================================
# MODEL & EVALUATION
# ============================================================

elif page == "Model & Evaluation":

    st.markdown(
        '<div class="section-title">'
        'Model & Evaluation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                Active Model
            </div>
            <div class="muted">
                YOLO11n V2-A · Fire / Smoke
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        'Measured Results'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        display_kpi(
            "MultiFire precision",
            "71.96%",
        )

    with c2:

        display_kpi(
            "MultiFire recall",
            "42.26%",
        )

    with c3:

        display_kpi(
            "MultiFire mAP50",
            "43.16%",
        )

    with c4:

        display_kpi(
            "MultiFire mAP50-95",
            "27.64%",
        )

    st.markdown(
        '<div class="section-title">'
        'Cross-domain image results'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        display_kpi(
            "Urban fire detection",
            "97.52%",
        )

    with c2:

        display_kpi(
            "Urban normal fire FAR",
            "0.89%",
        )

    with c3:

        display_kpi(
            "Any detection on normal",
            "1.28%",
        )

    st.info(
        "These are measured V2-A evaluation results. "
        "Object-level MultiFire metrics and image-level "
        "cross-domain metrics are different measurements."
    )


# ============================================================
# EVIDENCE
# ============================================================

elif page == "Evidence":

    st.markdown(
        '<div class="section-title">'
        'Evaluation Evidence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-note">'
        'Examples are from the existing V2-A evaluation artifacts.'
        '</div>',
        unsafe_allow_html=True,
    )

    cases = [
        (
            "case_01_success.jpg",
            "Case 1 — Successful Fire Detection",
        ),
        (
            "case_02_localization.jpg",
            "Case 2 — Detection with Localization Error",
        ),
        (
            "case_03_smoke_edge.jpg",
            "Case 3 — Smoke-Heavy Edge Case",
        ),
    ]

    for filename, title in cases:

        path = (
            EVIDENCE_DIR
            / filename
        )

        if path.exists():

            st.markdown(
                f'<div class="panel-title">'
                f'{title}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.image(
                str(path),
                width="stretch",
            )

        else:

            st.warning(
                f"Evidence file not found: {filename}"
            )


# ============================================================
# SYSTEM
# ============================================================

else:

    st.markdown(
        '<div class="section-title">'
        'System Status'
        '</div>',
        unsafe_allow_html=True,
    )

    import platform

    try:

        import torch

        torch_version = (
            torch.__version__
        )

        cuda_status = (
            "Available"
            if torch.cuda.is_available()
            else "CPU mode"
        )

    except Exception as exc:

        torch_version = (
            f"Unavailable: {exc}"
        )

        cuda_status = "Unavailable"

    c1, c2, c3 = st.columns(3)

    with c1:

        display_kpi(
            "Python",
            platform.python_version(),
        )

    with c2:

        display_kpi(
            "PyTorch",
            torch_version,
        )

    with c3:

        display_kpi(
            "Compute",
            cuda_status,
        )

    st.markdown(
        '<div class="section-title">'
        'Model File'
        '</div>',
        unsafe_allow_html=True,
    )

    st.code(
        str(MODEL_PATH)
    )

    st.write(
        "Model exists:",
        MODEL_PATH.exists(),
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SentinelEye Operations Console ·
        YOLO11n V2-A · Fire / Smoke Detection
    </div>
    """,
    unsafe_allow_html=True,
)
