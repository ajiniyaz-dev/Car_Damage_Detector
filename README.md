# Car Damage Detector

A web-based car damage detection demo powered by YOLOv8n. Upload a vehicle image, run inference, and view bounding boxes, confidence scores, and severity labels in a clean browser interface.

## Project Overview

This application detects visible car damage from uploaded images using a custom-trained YOLOv8n model. The workflow is:

```
Upload Image → YOLOv8n → Damage Detection → Bounding Boxes → Confidence Scores → Result Display
```

The backend exposes a FastAPI REST API. The frontend is a lightweight HTML/CSS/JavaScript client with no framework dependencies.

## Features

- Image upload with instant original-image preview
- YOLOv8n object detection with annotated result image
- Damage summary with colored status badges
- Per-detection confidence scores formatted as percentages
- Severity labels: High, Medium, Low
- Loading state during inference
- Responsive card-based layout
- Model information panel

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn |
| ML Model | YOLOv8n (Ultralytics) |
| Image Processing | OpenCV |
| Frontend | HTML5, CSS3, JavaScript |
| API Format | JSON + multipart file upload |

## YOLOv8n Model

- **Model file:** `models/yolo_damage_detector.pt`
- **Task:** Car damage detection
- **Framework:** Ultralytics YOLO
- **Confidence threshold:** 15%

Detections below the threshold are filtered by YOLO during inference. Returned confidence values are rounded to three decimal places and displayed as percentages in the UI.

### Severity Mapping

| Confidence | Severity |
|------------|----------|
| ≥ 50% | High |
| ≥ 30% | Medium |
| < 30% | Low |

## Project Structure

```
Car_Damage_Detector/
├── backend/
│   ├── app.py              # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Main page
│   ├── style.css           # UI styles
│   └── script.js           # Frontend logic
├── models/
│   └── yolo_damage_detector.pt
├── .vscode/
│   └── settings.json       # Live Server workspace settings
└── README.md
```

Runtime upload and prediction images are stored outside the project directory in:

`%LOCALAPPDATA%\car_damage_detector\`

## Installation

### Prerequisites

- Python 3.10+
- Git (optional)

### Setup

1. Clone or download the project.

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Install backend dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

4. Confirm the model file exists:

   ```text
   models/yolo_damage_detector.pt
   ```

## How To Run

### 1. Start the backend

```bash
cd backend
uvicorn app:app --reload
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start the frontend

Open `frontend/index.html` with Live Server, or serve the folder with any static file server:

```bash
cd frontend
python -m http.server 5502
```

Open: [http://127.0.0.1:5502/index.html](http://127.0.0.1:5502/index.html)

### 3. Configure API URL for deployment

Update the API base URL in `frontend/index.html`:

```html
<meta name="api-base" content="https://your-api-domain.com">
```

## Example Usage

1. Open the frontend in your browser.
2. Click **Choose Image** and select a car photo.
3. Review the **Original Image** preview.
4. Click **Detect Damage**.
5. View:
   - **Detection Result** with YOLO bounding boxes
   - **Analysis** with damage count, confidence, and severity
   - **Model Information** panel

### Example API Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@car.jpg"
```

### Example API Response

```json
{
  "damage_found": true,
  "damage_count": 2,
  "detections": [
    { "confidence": 0.619, "severity": "High" },
    { "confidence": 0.287, "severity": "Low" }
  ],
  "prediction_image_url": "/predictions/uuid_car.jpg",
  "confidence_threshold": 0.15
}
```

## Screenshots

<!-- Add screenshots here after deployment -->
| Screen | Description |
|--------|-------------|
| Upload | Image selection and detect button |
| Detection Result | YOLO annotated output |
| Analysis | Confidence and severity breakdown |

## Future Improvements

- Docker deployment for backend and frontend
- Environment-variable-based API configuration
- Authentication and rate limiting for public deployment
- Damage class labels per bounding box
- Batch image processing
- Exportable PDF inspection report

## License

Academic / demonstration project.
