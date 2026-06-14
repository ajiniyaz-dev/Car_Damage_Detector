"""
Parity test: compare ONNX Runtime inference against the original .pt model.

Requires ultralytics + torch in the local development environment.
Production requirements.txt does not include those packages.

Usage:
    python test_onnx_parity.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
PT_MODEL = MODELS_DIR / "yolo_damage_detector.pt"
ONNX_MODEL = MODELS_DIR / "yolo_damage_detector.onnx"

CONFIDENCE_THRESHOLD = 0.15
YOLO_IMGSZ = 640
CONFIDENCE_TOLERANCE = 0.02


def find_sample_images() -> list[Path]:
    """Collect sample images from local runtime uploads and synthetic cases."""

    samples: list[Path] = []
    uploads_dir = Path(
        os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
    ) / "car_damage_detector" / "uploads"

    if uploads_dir.is_dir():
        for path in sorted(uploads_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if path.name.endswith(".prepared.jpg"):
                continue
            samples.append(path)

    synthetic_dir = Path(tempfile.gettempdir()) / "car_damage_parity_samples"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    synthetic_cases = {
        "small_800x600.jpg": (800, 600),
        "medium_1280x720.jpg": (1280, 720),
        "large_4032x3024.jpg": (4032, 3024),
    }

    for filename, size in synthetic_cases.items():
        path = synthetic_dir / filename
        if not path.exists():
            Image.new("RGB", size, (140, 140, 140)).save(path, "JPEG")
        samples.append(path)

    return samples[:8]


def apply_prepare_image(image_path: Path) -> Path:
    """Mirror app.prepare_image_for_inference without importing FastAPI app."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    longest_side = max(width, height)
    max_dimension = 1280

    if longest_side <= max_dimension:
        return image_path

    scale = max_dimension / longest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    prepared_path = image_path.with_suffix(".prepared.jpg")
    cv2.imwrite(str(prepared_path), resized)
    return prepared_path


def run_pt_inference(image_path: Path) -> list[float]:
    from ultralytics import YOLO

    model = YOLO(str(PT_MODEL))
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        save=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    confidences = [round(float(box.conf[0]), 3) for box in boxes]
    confidences.sort(reverse=True)
    return confidences


def run_ultralytics_onnx_inference(image_path: Path) -> list[float]:
    from ultralytics import YOLO

    model = YOLO(str(ONNX_MODEL))
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        save=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    confidences = [round(float(box.conf[0]), 3) for box in boxes]
    confidences.sort(reverse=True)
    return confidences


def run_onnx_inference(image_path: Path) -> list[float]:
    sys.path.insert(0, str(BACKEND_DIR))
    from onnx_inference import YoloOnnxDetector

    detector = YoloOnnxDetector(
        str(ONNX_MODEL),
        conf_threshold=CONFIDENCE_THRESHOLD,
        imgsz=YOLO_IMGSZ
    )
    confidences, _ = detector.detect(str(image_path))
    return [round(float(value), 3) for value in confidences]


def compare_confidence_lists(
    pt_values: list[float],
    onnx_values: list[float]
) -> dict:
    """Compare detection counts and confidence values."""

    count_match = len(pt_values) == len(onnx_values)
    pairwise_matches = []

    if count_match and pt_values:
        for pt_value, onnx_value in zip(pt_values, onnx_values):
            pairwise_matches.append(
                abs(pt_value - onnx_value) <= CONFIDENCE_TOLERANCE
            )

    return {
        "pt_count": len(pt_values),
        "onnx_count": len(onnx_values),
        "count_match": count_match,
        "pt_confidences": pt_values,
        "onnx_confidences": onnx_values,
        "confidence_match": (
            count_match
            and (
                not pt_values
                or all(pairwise_matches)
            )
        ),
    }


def measure_memory() -> dict:
    """Measure isolated process memory for PT app vs ONNX app."""

    import subprocess

    python_executable = sys.executable
    measure_script = f"""
import json, os, sys, gc, tempfile, psutil
from pathlib import Path

root = {json.dumps(str(PROJECT_ROOT))}
backend = {json.dumps(str(BACKEND_DIR))}
proc = psutil.Process(os.getpid())

def rss():
    gc.collect()
    return round(proc.memory_info().rss / (1024 * 1024), 1)

mode = sys.argv[1]
out = {{"mode": mode, "baseline_mb": rss()}}

if mode == "onnx_app":
    sys.path.insert(0, backend)
    import app
    out["after_load_mb"] = rss()
elif mode == "pt_app":
    sys.path.insert(0, backend)
    os.environ["MODEL_PATH"] = str(Path(root) / "models" / "yolo_damage_detector.pt")
    from ultralytics import YOLO
    model = YOLO(os.environ["MODEL_PATH"])
    out["after_load_mb"] = rss()

print(json.dumps(out))
"""

    results = {}
    for mode in ("onnx_app", "pt_app"):
        completed = subprocess.run(
            [python_executable, "-c", measure_script, mode],
            capture_output=True,
            text=True,
            check=False
        )
        if completed.returncode != 0:
            results[mode] = {"error": completed.stderr.strip()}
            continue
        results[mode] = json.loads(completed.stdout.strip())

    return results


def main() -> int:
    if not PT_MODEL.is_file():
        print(f"Missing PT model: {PT_MODEL}")
        return 1

    if not ONNX_MODEL.is_file():
        print(f"Missing ONNX model: {ONNX_MODEL}")
        return 1

    samples = find_sample_images()
    if not samples:
        print("No sample images found for parity testing.")
        return 1

    report = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "imgsz": YOLO_IMGSZ,
        "confidence_tolerance": CONFIDENCE_TOLERANCE,
        "images_tested": [],
        "summary": {
            "total_images": 0,
            "pt_count_matches": 0,
            "pt_confidence_matches": 0,
            "onnx_runtime_matches": 0,
            "parity_passed": False,
        },
        "memory": measure_memory(),
    }

    for sample in samples:
        prepared = apply_prepare_image(sample)
        pt_values = run_pt_inference(prepared)
        reference_onnx_values = run_ultralytics_onnx_inference(prepared)
        onnx_values = run_onnx_inference(prepared)

        pt_comparison = compare_confidence_lists(pt_values, onnx_values)
        runtime_comparison = compare_confidence_lists(
            reference_onnx_values,
            onnx_values
        )

        comparison = {
            "image": str(sample),
            "pt_confidences": pt_values,
            "ultralytics_onnx_confidences": reference_onnx_values,
            "our_onnx_confidences": onnx_values,
            "pt_vs_our_onnx": pt_comparison,
            "ultralytics_onnx_vs_our_onnx": runtime_comparison,
        }
        report["images_tested"].append(comparison)

        if pt_comparison["count_match"]:
            report["summary"]["pt_count_matches"] += 1
        if pt_comparison["confidence_match"]:
            report["summary"]["pt_confidence_matches"] += 1
        if runtime_comparison["confidence_match"]:
            report["summary"]["onnx_runtime_matches"] += 1

    report["summary"]["total_images"] = len(samples)
    report["summary"]["parity_passed"] = (
        report["summary"]["onnx_runtime_matches"] == len(samples)
    )

    print(json.dumps(report, indent=2))
    return 0 if report["summary"]["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
