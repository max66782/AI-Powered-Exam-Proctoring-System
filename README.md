# AI-Powered Exam Proctoring System

A real-time AI-based exam proctoring system that monitors candidate behavior during online examinations using Computer Vision, Deep Learning, Face Recognition, and Object Detection techniques.

The system automatically detects suspicious activities such as phone usage, multiple person presence, candidate absence, head movement, and unauthorized person substitution. It also generates evidence snapshots, event logs, risk scores, and session reports for audit purposes.

---

## Demo

![AI Proctor Demo](docs/demo.png)

---

## Features

### Candidate Verification
- Face verification using InsightFace embeddings
- Detection of unauthorized candidates
- Identity matching against registered student embeddings

### Phone Detection
- Real-time mobile phone detection using YOLOv8
- Confidence-based detection filtering
- Evidence capture on violation

### Multiple Person Detection
- Detects more than one person in front of the webcam
- Logs violations after configurable duration

### Face Missing Detection
- Detects prolonged absence of the candidate
- Generates evidence and logs suspicious behavior

### Head Direction Monitoring
- Tracks head orientation using MediaPipe Face Mesh
- Detects prolonged looking-away behavior

### Risk Scoring Engine
- Assigns risk points for suspicious activities
- Generates cumulative exam risk score

### Evidence Capture
- Stores screenshots when violations occur
- Maintains an audit trail for review

### Session Reporting
- Generates exam summary reports
- Includes event counts and final risk score

### Performance Monitoring
- Real-time FPS display
- Optimized identity verification pipeline

---

## System Architecture

```text
Webcam Input
      │
      ▼
YOLOv8 Object Detection
      │
      ▼
MediaPipe Face Detection & Face Mesh
      │
      ▼
InsightFace Verification
      │
      ▼
Violation Detection Engine
      │
      ▼
Risk Scoring Engine
      │
      ▼
Evidence Capture + Event Logging + Session Reports
```

---

## Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- InsightFace
- NumPy
- SciPy
- ONNX Runtime
- YAML Configuration

---

## Project Structure

```text
AI-Proctor/
│
├── src/
│   ├── proctor_engine.py
│   ├── phone_detection.py
│   ├── face_verification.py
│   └── ...
│
├── data/
│   └── embeddings/
│
├── docs/
│   └── demo.png
│
├── evidence/
├── logs/
├── reports/
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/max66782/AI-Proctor.git
cd AI-Proctor
```

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python src/proctor_engine.py
```

---

## Current Capabilities

| Module | Status |
|----------|----------|
| Face Verification | ✅ |
| Phone Detection | ✅ |
| Multiple Person Detection | ✅ |
| Face Missing Detection | ✅ |
| Head Direction Detection | ✅ |
| Risk Scoring | ✅ |
| Evidence Capture | ✅ |
| Session Reports | ✅ |
| FPS Monitoring | ✅ |
| Configurable Thresholds | ✅ |

---

## Future Enhancements

- Eye Gaze Tracking
- Mouth Activity Detection
- Audio Monitoring
- Dashboard Analytics
- Database Integration
- Cloud Deployment
- Multi-Camera Support

---

## Author

**Krishn Bhushan Singh**  
Electrical Engineering, IIT Roorkee

Interested in Software Engineering, Data Engineering, Computer Vision, Machine Learning, and Artificial Intelligence.