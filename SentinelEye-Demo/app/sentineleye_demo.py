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
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- PAGE ---------- */

    .stApp {
        background: #f4f6f8;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    /* ---------- HEADER ---------- */

    .topbar {
        background: #ffffff;
        border: 1px solid #d9dee5;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }

    .title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #17324d;
        margin: 0;
    }

    .subtitle {
        font-size: 0.82rem;
        color: #687586;
        margin-top: 2px;
    }

    .system-status {
        text-align: right;
        color: #19703b;
        font-weight: 600;
        font-size: 0.85rem;
        padding-top: 8px;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        color: #17324d;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .section-note {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: -5px;
        margin-bottom: 12px;
    }

    /* ---------- KPI ---------- */

    .kpi {
        background: #ffffff;
        border: 1px solid #d9dee5;
        border-radius: 6px;
        padding: 14px 16px;
        min-height: 92px;
    }

    .kpi-label {
        color: #6b7280;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .kpi-value {
        color: #17324d;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .kpi-danger {
        color: #b42318;
    }

    .kpi-warning {
        color: #a15c00;
    }

    .kpi-success {
        color: #19703b;
    }

    /* ---------- PANELS ---------- */

    .panel {
        background: #ffffff;
        border: 1px solid #d9dee5;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: #17324d;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .muted {
        color: #6b7280;
        font-size: 0.80rem;
    }

    /* ---------- ALERTS ---------- */

    .alert-fire {
        border-left: 4px solid #b42318;
        background: #fff4f2;
        padding: 11px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }

    .alert-smoke {
        border-left: 4px solid #a15c00;
        background: #fff8ec;
        padding: 11px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }

    .alert-safe {
        border-left: 4px solid #19703b;
        background: #f1faf4;
        padding: 11px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        border-top: 1px solid #d9dee5;
        margin-top: 20px;
        padding-top: 10px;
        color: #7a8592;
        font-size: 0.72rem;
        text-align: center;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 800px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 0.7rem;
        }

        .title {
            font-size: 1.3rem;
        }

        .system-status {
            text-align: left;
            padding-top: 3px;
        }

        .kpi {
            min-height: 78px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "incidents" not in st.session_state:
    st.session_state.incidents = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_time" not in st.session_state:
    st.session_state.last_time = None


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

        label = (
            f"{name} "
            f"{score:.2f}"
        )

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


def get_status(
    fire_count,
    smoke_count,
):
    if fire_count > 0:
        return (
            "FIRE ALERT",
            "fire",
        )

    if smoke_count > 0:
        return (
            "SMOKE DETECTED",
            "smoke",
        )

    return (
        "NO THREAT",
        "safe",
    )


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

    # Keep session log manageable.
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
            <div class="kpi-label">
                {label}
            </div>
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

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:
    st.markdown(
        """
        <div class="topbar">
            <div class="title">
                SentinelEye Operations
            </div>
            <div class="subtitle">
                UAV-based fire and smoke monitoring
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div class="topbar">
            <div class="system-status">
                ● SYSTEM ONLINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### SentinelEye"
    )

    page = st.radio(
        "Operations",
        [
            "Overview",
            "Live Detection",
            "Alerts",
            "Model & Evaluation",
            "Evidence",
            "System",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONF,
        step=0.05,
    )

    st.divider()

    st.caption("MODEL")
    st.write("YOLO11n V2-A")
    st.write("Classes: Fire / Smoke")
    st.write("Input: 640 px")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Operations Overview</div>',
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
        '<div class="section-title">Current Monitoring</div>',
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
                    Upload an image or use the camera
                    from the Live Detection section.
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
                use_container_width=True,
            )

        else:
            st.info(
                "No detection has been run in this session."
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

            status, status_class = (
                get_status(
                    latest["fire"],
                    latest["smoke"],
                )
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
        '<div class="section-title">Recent Incidents</div>',
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
                    "Confidence": item[
                        "confidence"
                    ],
                    "Inference (ms)": item[
                        "inference_ms"
                    ],
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
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
        '<div class="section-title">Live Detection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-note">'
        'Run the real V2-A model on an image, camera frame, or video.'
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

            c1, c2, c3, c4 = (
                st.columns(4)
            )

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
                use_container_width=True,
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
                            "Class": item[
                                "class"
                            ],
                            "Confidence": round(
                                item[
                                    "confidence"
                                ],
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
                    use_container_width=True,
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
                    use_container_width=True,
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
        )

        if uploaded:

            max_frames = st.slider(
                "Maximum frames to process",
                min_value=30,
                max_value=600,
                value=180,
                step=30,
            )

            suffix = Path(
                uploaded.name
            ).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp:

                temp.write(
                    uploaded.getbuffer()
                )

                video_path = temp.name

            cap = cv2.VideoCapture(
                video_path
            )

            if not cap.isOpened():
                st.error(
                    "Unable to open this video."
                )

            else:

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                fps = (
                    cap.get(
                        cv2.CAP_PROP_FPS
                    )
                    or 25
                )

                image_slot = st.empty()
                status_slot = st.empty()
                progress_slot = st.progress(
                    0.0
                )

                processed = 0

                while (
                    processed < max_frames
                ):

                    ok, frame = (
                        cap.read()
                    )

                    if not ok:
                        break

                    result, inference_ms = (
                        run_detection(
                            frame,
                            confidence,
                        )
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

                    annotated_rgb = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB,
                    )

                    image_slot.image(
                        annotated_rgb,
                        use_container_width=True,
                    )

                    status, _ = (
                        get_status(
                            fire_count,
                            smoke_count,
                        )
                    )

                    status_slot.write(
                        f"**{status}**  |  "
                        f"Fire: **{fire_count}**  |  "
                        f"Smoke: **{smoke_count}**  |  "
                        f"Inference: "
                        f"**{inference_ms:.0f} ms**"
                    )

                    processed += 1

                    if total_frames > 0:
                        progress_slot.progress(
                            min(
                                1.0,
                                processed
                                / min(
                                    total_frames,
                                    max_frames,
                                ),
                            )
                        )

                cap.release()

                st.success(
                    f"Processed {processed} frames "
                    f"from {uploaded.name}."
                )


# ============================================================
# ALERTS
# ============================================================

elif page == "Alerts":

    st.markdown(
        '<div class="section-title">Alerts & Incident Log</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.incidents:

        st.success(
            "No alerts have been generated in this session."
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
            if x["status"]
            == "SMOKE DETECTED"
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
                    "Confidence": item[
                        "confidence"
                    ],
                    "Inference (ms)": item[
                        "inference_ms"
                    ],
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
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
        '<div class="section-title">Model & Evaluation</div>',
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
        '<div class="section-title">Measured Results</div>',
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
        '<div class="section-title">Cross-domain image results</div>',
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
        "These figures are measured V2-A evaluation results. "
        "Object-level MultiFire metrics and image-level "
        "cross-domain metrics are different measurements."
    )


# ============================================================
# EVIDENCE
# ============================================================

elif page == "Evidence":

    st.markdown(
        '<div class="section-title">Evaluation Evidence</div>',
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
                f'<div class="panel-title">{title}</div>',
                unsafe_allow_html=True,
            )

            st.image(
                str(path),
                use_container_width=True,
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
        '<div class="section-title">System Status</div>',
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
        '<div class="section-title">Model File</div>',
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