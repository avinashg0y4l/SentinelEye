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

### MultiFire20K

MultiFire20K is a UAV-based fire monitoring dataset containing urban and rural fire imagery.

For the current V2-A experiment, the project used manual fire annotations from the official training split and a held-out urban-fire evaluation set.

> MultiFire manual supervision used in this experiment provides Fire annotations. Therefore Smoke performance should not be interpreted as fully validated against MultiFire Fire-only ground truth.

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

- Images: 1,010
- Ground-truth Fire boxes: 2,572

| Metric | Result |
|---|---:|
| Precision | 71.96% |
| Recall | 42.26% |
| mAP50 | 43.16% |
| mAP50-95 | 27.64% |

### Cross-domain image-level evaluation

#### Urban Fire

- Images: 1,010
- Fire detection rate: **97.52%**
- Any detection rate: **98.42%**

#### Urban Normal

- Images: 1,014
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

---

## 9. Known Limitations

### Localization

The model can correctly identify a fire event while producing a bounding box that is broader or less precise than the manual annotation.

### Smoke-heavy scenes

Large smoke plumes can make precise localization difficult.

### MultiFire annotation limitation

The MultiFire manual training/evaluation data used in this experiment provides Fire supervision, so Smoke performance must not be interpreted as a fully validated MultiFire Smoke result.

---

## 10. How to Run Locally

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

---

## 11. Evidence

Curated submission evidence is stored in:

```text
results/submission/visual_cases/
```

The evidence includes:

1. Successful Fire detection
2. Fire detection with localization limitation
3. Smoke-heavy edge case

---

## 12. Future Scope

Future development can include:

- improved Fire localization,
- stronger Smoke supervision,
- sequence-aware evaluation across more sources,
- drone telemetry integration,
- geospatial incident mapping,
- long-running alert management,
- centralized multi-camera monitoring,
- deployment on edge hardware,
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
```

SentinelEye is currently a working research/prototype system focused on measurable UAV-based fire and smoke detection.
