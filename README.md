# SentinelEye

## Fire & Smoke Detection for UAV / Smart-City Monitoring

SentinelEye is a computer-vision prototype for detecting **Fire** and **Smoke** in UAV and smart-city monitoring imagery.

The system combines a YOLO11n object detector with an operations-oriented dashboard for image, camera, and video inference.

---

## 1. Problem

Early detection of fire from aerial and urban imagery can support faster situational awareness and incident response.

SentinelEye is designed to:

- detect visible Fire and Smoke regions,
- display bounding boxes and confidence scores,
- provide an operator-oriented monitoring interface,
- preserve measurable evaluation evidence.

---

## 2. What SentinelEye Does

The current working prototype provides:

- Fire detection
- Smoke detection
- Bounding-box visualization
- Confidence scores
- Image inference
- Browser camera inference
- Video inference
- Session-based incident logging
- Operations-style dashboard
- Model and evaluation summary
- Presentation evidence cases

The system uses the trained **SentinelEye V2-A** detector.

---

## 3. System Architecture

```text
Input
 ├── Image
 ├── Camera frame
 └── Video
        │
        ▼
   YOLO11n V2-A
        │
        ├── Fire
        └── Smoke
        │
        ▼
Bounding Boxes + Confidence
        │
        ▼
SentinelEye Operations Dashboard
```

---

## 4. Model

### Detector

- Architecture: YOLO11n
- Version: SentinelEye V2-A
- Classes:
  - `0 = Smoke`
  - `1 = Fire`
- Inference image size: 640 × 640

### Model file

```text
SentinelEye-Demo/models/sentineleye_v2a.pt
```

---

## 5. Datasets

### D-Fire

D-Fire provides Fire and Smoke object-detection annotations in YOLO format.

The project uses D-Fire for detector training and validation.

**Source:**  
https://github.com/gaia-solutions-on-demand/DFireDataset

### MultiFire20K

MultiFire20K is a UAV-based fire monitoring dataset containing urban and rural fire imagery.

For the current V2-A experiment, the project used manual Fire annotations from the official training split and a held-out urban-fire evaluation set.

> MultiFire manual supervision used in this experiment provides Fire annotations. Therefore Smoke performance should not be interpreted as fully validated against MultiFire Fire-only ground truth.

**Source:**  
https://zenodo.org/records/17047113

---

## 6. Training Approach

### V1

The first detector used YOLO11n with D-Fire.

### V2-A

The current submission model combines:

- D-Fire detection supervision
- manually annotated MultiFire20K Fire data
- sequence-aware MultiFire train/validation preparation
- held-out MultiFire urban testing

The final submission model is the V2-A checkpoint.

---

## 7. Measured Results

### MultiFire Urban Fire Object Evaluation

- Images: **1,010**
- Ground-truth Fire boxes: **2,572**

| Metric | Result |
|---|---:|
| Precision | 71.96% |
| Recall | 42.26% |
| mAP50 | 43.16% |
| mAP50-95 | 27.64% |

### Cross-domain image-level evaluation

#### Urban Fire

- Images: **1,010**
- Fire detection rate: **97.52%**
- Any detection rate: **98.42%**

#### Urban Normal

- Images: **1,014**
- Fire false-alarm rate: **0.89%**
- Smoke false-alarm rate: **0.39%**
- Any detection rate: **1.28%**

The image-level detection results and object-level localization metrics measure different aspects of the model.

---

## 8. Live Dashboard

SentinelEye includes a Streamlit-based operations dashboard.

### Current dashboard sections

- Operations Overview
- Live Detection
- Alerts
- Model & Evaluation
- Evidence
- System

### Supported inputs

```text
Image
Camera
Video
```

The interface is designed as an internal operations console rather than a consumer-facing application.

### Dashboard

Public working application:

https://sentineleye-avinash.streamlit.app

### ELCIA Demonstration Video

https://www.youtube.com/watch?v=gk-8vqSjHsM

---

## 9. Testing Video

A sample test video can be used to verify the SentinelEye video-inference pipeline.

### How to test

Open the SentinelEye dashboard:

```text
Live Detection
        ↓
Video
        ↓
Upload the test video
```

The video can be used to test:

- frame-by-frame Fire/Smoke inference
- bounding-box visualization
- confidence scoring
- incident logging
- processed-video generation

### Test video

**Testing video:**

The demonstration video used for the ELCIA submission is available here:

https://www.youtube.com/watch?v=gk-8vqSjHsM

For independent functional testing, use a separate sample input video and upload it through:

```text
Live Detection
        ↓
Video
        ↓
Upload the test video
```

For independent testing, the Google Drive file should be shared as:

```text
General access → Anyone with the link
Role → Viewer
```

For independent testing, the Google Drive file should be shared as:

```text
General access → Anyone with the link
Role → Viewer
```

---

## 10. Known Limitations

### Localization

The model can correctly identify a fire event while producing a bounding box that is broader or less precise than the manual annotation.

### Smoke-heavy scenes

Large smoke plumes can make precise localization difficult.

### MultiFire annotation limitation

The MultiFire manual training/evaluation data used in this experiment provides Fire supervision, so Smoke performance must not be interpreted as a fully validated MultiFire Smoke result.

### Real-world smoke context

The current visual detector does not yet reliably distinguish hazardous fire-related smoke from normal industrial emissions such as brick-kiln or factory-chimney exhaust, construction dust, steam, or other smoke-like sources.

This is a known prototype limitation and is a focus for future contextual risk analysis.

---

## 11. How to Run Locally

### Requirements

- Python 3.12+
- PyTorch
- Ultralytics
- Streamlit
- OpenCV
- Pillow
- NumPy

### Start the dashboard

From:

```text
SentinelEye/SentinelEye-Demo
```

activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```powershell
python -m streamlit run .pp\sentineleye_demo.py
```

Open:

```text
http://localhost:8501
```

### Dependency installation

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

---

## 12. Evidence

Curated submission evidence is stored in:

```text
results/submission/visual_cases/
```

Additional dashboard evidence is stored in:

```text
SentinelEye-Demo/evidence/
```

The evidence includes:

1. Successful Fire detection
2. Fire detection with localization limitation
3. Smoke-heavy edge case

---

## 13. Future Scope

The current prototype is intentionally focused on the detection and operations layer.

Future development can include:

- improved Fire localization,
- stronger Smoke supervision,
- classification of different smoke sources,
- temporal analysis across consecutive video frames,
- risk scoring and Low/Medium/High/Critical incident classification,
- geospatial incident mapping,
- persistent incident management,
- drone GPS and telemetry integration,
- centralized multi-drone monitoring,
- cross-drone incident verification,
- thermal and environmental sensor fusion,
- configurable authority escalation,
- site-specific industrial-emission baselines,
- deployment on edge hardware,
- continuous model retraining using additional real-world data,
- integration with operational command systems.

These features are future scope unless explicitly demonstrated in the current build.

---

## Current Status

### Working

- YOLO11n V2-A inference
- Fire detection
- Smoke detection
- Bounding boxes
- Confidence scores
- Image inference
- Camera inference
- Video inference
- Operations dashboard
- Session-based incident logging
- Evaluation evidence

### Measured

- D-Fire detector evaluation
- MultiFire urban object evaluation
- Urban fire image-level evaluation
- Urban normal false-alarm evaluation

### Not claimed as implemented

- Real drone control
- Live GPS/telemetry integration
- Production emergency dispatch
- Large-scale cloud orchestration
- Fully autonomous emergency response
- Automatic authority contact based only on a Smoke/Fire prediction
- Production-grade multi-drone coordination

---

## Repository

GitHub:

https://github.com/avinashg0y4l/SentinelEye

---

## Final Model

```text
SentinelEye V2-A
YOLO11n
Classes: Smoke, Fire
Input size: 640 × 640
```

SentinelEye is currently a working research/prototype system focused on measurable UAV-based Fire and Smoke detection.

The project is structured to provide a clear path from visual detection toward a future contextual, risk-aware and multi-drone smart-city monitoring platform.
