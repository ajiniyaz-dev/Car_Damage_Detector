from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from onnx_inference import YoloOnnxDetector
import shutil
import os
import uuid
import cv2
import tempfile

CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.7
MAX_INFERENCE_DIMENSION = 1280
YOLO_IMGSZ = 640

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        PROJECT_ROOT,
        "models",
        "yolo_damage_detector.onnx"
    )
)

_DATA_ROOT = os.environ.get(
    "DATA_DIR",
    os.path.join(
        tempfile.gettempdir(),
        "car_damage_detector"
    )
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
        f"YOLO ONNX model not found at {MODEL_PATH}"
    )

detector = YoloOnnxDetector(
    MODEL_PATH,
    conf_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
    imgsz=YOLO_IMGSZ
)


def get_severity(confidence: float) -> str:

    if confidence >= 0.50:
        return "High"

    if confidence >= 0.30:
        return "Medium"

    return "Low"


def prepare_image_for_inference(image_path: str) -> None:
    """
    Downscale very large uploads before YOLO inference.

    Phone and camera photos (3000x4000, 4K, etc.) allocate large OpenCV
    and PyTorch tensors during decode, inference, and annotation. Capping the
    longest side at 1280 px preserves aspect ratio, keeps dent detail usable,
    and lowers peak RAM on memory-limited hosts such as Render Free (512 MB).
    """

    image = cv2.imread(image_path)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image file"
        )

    original_height, original_width = image.shape[:2]
    print(
        f"Original image: {original_width}x{original_height}"
    )

    longest_side = max(
        original_width,
        original_height
    )

    if longest_side <= MAX_INFERENCE_DIMENSION:
        print(
            "Image size acceptable, no resize performed."
        )
        return

    scale = MAX_INFERENCE_DIMENSION / longest_side
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    # INTER_AREA is preferred when shrinking images to limit aliasing.
    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(
        image_path,
        resized_image
    )

    print(
        f"Resized image: {new_width}x{new_height}"
    )


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

    prepare_image_for_inference(image_path)

    try:
        confidences, annotated_image = detector.detect(image_path)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    detections = []

    for confidence in confidences:
        rounded_confidence = round(
            float(confidence),
            3
        )

        detections.append({
            "confidence": rounded_confidence,
            "severity": get_severity(rounded_confidence)
        })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

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
