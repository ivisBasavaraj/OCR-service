"""
Scale OCR Microservice — FastAPI
Uses original YOLOv8 digit-detection model (best.pt) with
advanced image preprocessing to boost accuracy.
"""

import os
import re
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2

app = FastAPI(title="Scale OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# class index → character
CLASS_MAP = {
    0: '-',
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
    6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: '.',
}

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).parent / "best.pt"))
model: YOLO | None = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/read-scale")
async def read_scale(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="OCR model not loaded")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Try multiple preprocessed versions and pick best result
    reading, confidence = _read_with_best_preprocessing(img)

    if reading is None:
        raise HTTPException(status_code=422, detail="Could not detect digits on scale display")

    try:
        value = float(reading)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Non-numeric reading: '{reading}'")

    return {
        "reading": reading,
        "valueGrams": round(value, 3),
        "confidence": round(confidence, 3),
    }


# ── Preprocessing variants ────────────────────────────────────────────────────

def _preprocess_variants(img: np.ndarray):
    """Generate multiple preprocessed versions of the image."""
    variants = []

    # 1. Original (just resized)
    variants.append(("original", img.copy()))

    # 2. CLAHE — contrast limited adaptive histogram equalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    variants.append(("clahe", clahe_img))

    # 3. Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    variants.append(("sharp", sharp))

    # 4. Denoised + sharpened
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    sharp_denoised = cv2.filter2D(denoised, -1, kernel)
    variants.append(("denoise+sharp", sharp_denoised))

    # 5. Brightness boosted (for dark/underexposed images)
    bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
    variants.append(("bright", bright))

    # 6. CLAHE + sharpened
    clahe_sharp = cv2.filter2D(clahe_img, -1, kernel)
    variants.append(("clahe+sharp", clahe_sharp))

    return variants


# ── Core detection ────────────────────────────────────────────────────────────

def _read_with_best_preprocessing(img: np.ndarray):
    """
    Try all preprocessing variants, return the result with
    highest confidence and most digits detected.
    """
    variants = _preprocess_variants(img)
    best_reading = None
    best_conf = 0.0
    best_score = -1

    for name, processed in variants:
        reading, conf, n_digits = _yolo_read(processed)
        if reading is None:
            continue

        # Score = number of digits detected * confidence
        # Prefer readings with decimal point (more precise)
        has_dot = '.' in reading
        score = n_digits * conf * (1.2 if has_dot else 1.0)

        print(f"  [{name}] reading={reading} conf={conf:.3f} digits={n_digits} score={score:.3f}")

        if score > best_score:
            best_score = score
            best_reading = reading
            best_conf = conf

    return best_reading, best_conf


def _yolo_read(img: np.ndarray):
    """
    Run YOLO on image, sort boxes left-to-right, assemble numeric string.
    Returns (reading_str, avg_confidence, digit_count) or (None, 0.0, 0).
    """
    # Run at multiple confidence thresholds — lower conf catches faint digits
    for conf_thresh in [0.35, 0.25, 0.15]:
        results = model(img, verbose=False, conf=conf_thresh)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            continue

        detections = []
        for box in boxes:
            cls_id   = int(box.cls[0].item())
            conf     = float(box.conf[0].item())
            x_center = float(box.xywh[0][0].item())
            char = CLASS_MAP.get(cls_id)
            if char is not None:
                detections.append((x_center, char, conf))

        if not detections:
            continue

        # Remove duplicate detections at same x position (keep highest conf)
        detections = _deduplicate(detections)

        # Sort left-to-right
        detections.sort(key=lambda d: d[0])

        reading  = ''.join(d[1] for d in detections)
        avg_conf = sum(d[2] for d in detections) / len(detections)

        reading = _sanitise(reading)
        if reading is None:
            continue

        return reading, avg_conf, len(detections)

    return None, 0.0, 0


def _deduplicate(detections, overlap_px=15):
    """
    Remove duplicate boxes at nearly the same x position.
    Keep the one with highest confidence.
    """
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    for det in detections:
        x = det[0]
        if not any(abs(x - k[0]) < overlap_px for k in kept):
            kept.append(det)
    return kept


def _sanitise(raw: str) -> str | None:
    """Clean assembled string into a valid numeric reading."""
    # Remove leading/trailing non-numeric chars
    match = re.search(r'-?\d+\.?\d*', raw)
    if not match:
        return None
    val_str = match.group(0)
    # Validate range (gold weight: 0.001g to 9999g)
    try:
        val = float(val_str)
        if val < 0.001 or val > 9999.999:
            return None
    except ValueError:
        return None
    return val_str


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
