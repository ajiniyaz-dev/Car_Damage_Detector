from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
import uuid
import cv2

CONFIDENCE_THRESHOLD = 0.15

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolo_damage_detector.pt"
)

_DATA_ROOT = os.path.join(
    os.environ.get(
        "LOCALAPPDATA",
        os.environ.get("TEMP", ".")
    ),
    "car_damage_detector"
)

UPLOAD_FOLDER = os.path.join(
    _DATA_ROOT,
    "uploads"
)

PREDICTION_FOLDER = os.path.join(
    _DATA_ROOT,
    "predictions"
)

app = FastAPI(
    title="Car Damage Detector API",
    description="YOLOv8n-based car damage detection service"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PREDICTION_FOLDER,
    exist_ok=True
)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found at {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)


def get_severity(confidence: float) -> str:

    if confidence >= 0.50:
        return "High"

    if confidence >= 0.30:
        return "Medium"

    return "Low"


@app.get("/")
def root():

    return {
        "message": "Car Damage Detection API Running",
        "model": "YOLOv8n",
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.get("/predictions/{filename}")
async def get_prediction_image(filename: str):

    safe_name = os.path.basename(filename)

    file_path = os.path.join(
        PREDICTION_FOLDER,
        safe_name
    )

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="Prediction image not found"
        )

    return FileResponse(
        file_path,
        media_type="image/jpeg"
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )

    unique_name = (
        str(uuid.uuid4())
        + "_"
        + os.path.basename(file.filename)
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    with open(
        image_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        save=False
    )

    detections = []
    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            confidence = round(
                float(box.conf[0]),
                3
            )

            detections.append({
                "confidence": confidence,
                "severity": get_severity(confidence)
            })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    annotated_image = results[0].plot()

    prediction_path = os.path.join(
        PREDICTION_FOLDER,
        unique_name
    )

    cv2.imwrite(
        prediction_path,
        annotated_image
    )

    return {
        "damage_found": len(detections) > 0,
        "damage_count": len(detections),
        "detections": detections,
        "prediction_image_url": f"/predictions/{unique_name}",
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }
