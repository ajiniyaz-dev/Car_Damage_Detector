# 🚗 Car Damage Detector

## Overview

Car Damage Detector is a machine learning web application that automatically detects visible vehicle damage from uploaded images.

The system uses a custom-trained YOLOv8n object detection model that has been optimized and deployed using ONNX Runtime for efficient inference and reduced memory consumption.

Users can upload an image of a vehicle, and the application will:

* Detect visible dents or damaged regions
* Draw bounding boxes around detected damage
* Display confidence scores
* Classify detection severity
* Provide a visual analysis report

The application is deployed online and can be accessed through a web browser without installing any software.

---

## Live Demo

### Frontend

https://car-damage-detector-gamma.vercel.app/

### Backend API

https://car-damage-detector-5z60.onrender.com/

### API Documentation

https://car-damage-detector-5z60.onrender.com/docs

---

## Features

### Image Upload

Users can upload vehicle images directly through the web interface.

### Damage Detection

The model automatically identifies visible damage regions.

### Bounding Box Visualization

Detected damage areas are highlighted with bounding boxes.

### Confidence Scores

Each detection includes a confidence percentage.

### Severity Classification

Detected damages are categorized into:

* High Severity
* Medium Severity
* Low Severity

### Responsive User Interface

The application works on desktop and mobile devices.

### Cloud Deployment

The project is publicly accessible through Vercel and Render.

---

## Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* FastAPI
* Python

### Machine Learning

* YOLOv8n
* ONNX Runtime
* OpenCV

### Deployment

* Vercel (Frontend)
* Render (Backend)

### Version Control

* Git
* GitHub

---

## System Architecture

```text
User Uploads Image
        │
        ▼
Frontend (Vercel)
        │
        ▼
FastAPI Backend (Render)
        │
        ▼
ONNX Runtime
        │
        ▼
Damage Detection
        │
        ▼
Annotated Image + Analysis
        │
        ▼
Displayed to User
```

---

## Machine Learning Model

### Model Type

YOLOv8n Object Detection Model

### Detection Class

| Class ID | Class Name |
| -------- | ---------- |
| 0        | Damage     |

### Model Information

* Model Architecture: YOLOv8n
* Inference Engine: ONNX Runtime
* Confidence Threshold: 0.15
* Image Size: 640 × 640
* Number of Classes: 1

### Optimization

The original PyTorch model was converted to ONNX Runtime to reduce memory usage and improve deployment stability.

#### Memory Comparison

| Stage                 | PyTorch | ONNX Runtime |
| --------------------- | ------- | ------------ |
| Startup Memory        | ~308 MB | ~92 MB       |
| Peak Inference Memory | ~390 MB | ~131 MB      |

This optimization reduced peak memory usage by approximately 66%.

---

## Project Structure

```text
Car_Damage_Detector/
│
├── backend/
│   ├── app.py
│   ├── onnx_inference.py
│   ├── requirements.txt
│   └── test_onnx_parity.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   ├── yolo_damage_detector.pt
│   └── yolo_damage_detector.onnx
│
├── screenshots/
│   ├── home-page.png
│   ├── image-uploaded.png
│   ├── damage-detected.png
│   ├── multiple-damages.png
│   └── live-demo.png
│
├── render.yaml
├── README.md
└── .gitignore
```

---

## Screenshots

### Home Page

![Home Page](screenshots/home-page.png)

---

### Image Uploaded

![Image Uploaded](screenshots/image-uploaded.png)

---

### Damage Detection Result

![Damage Detection](screenshots/damage-detected.png)

---

### Multiple Damage Detection

![Multiple Damage Detection](screenshots/multiple-damages.png)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/ajiniyaz-dev/Car_Damage_Detector.git
```

```bash
cd Car_Damage_Detector
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

### Run Backend

```bash
cd backend
uvicorn app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

### Run Frontend

Open:

```text
frontend/index.html
```

or use:

```bash
cd frontend
python -m http.server 5500
```

Frontend URL:

```text
http://127.0.0.1:5500
```

---

## Example Workflow

1. Upload a vehicle image.
2. Click "Detect Damage".
3. The image is sent to the backend.
4. ONNX Runtime performs object detection.
5. Damage regions are identified.
6. Bounding boxes are drawn.
7. Confidence scores and severity levels are calculated.
8. Results are displayed to the user.

---

## Future Improvements

Potential future enhancements include:

* Multiple damage categories
* Vehicle part detection
* Repair cost estimation
* Damage severity prediction using additional models
* Video-based damage detection
* User authentication and history tracking
* Cloud storage integration

---

## Author

**Ajiniyaz Bazarbaev**

Information Technology Student

Machine Learning and Artificial Intelligence Enthusiast

---

## License

This project is intended for educational, research, and portfolio purposes.
