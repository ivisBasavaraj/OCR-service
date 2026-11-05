"""
Scale OCR Microservice — FastAPI
Classes from data.yaml:
  0:-ve  1:No_0  2:No_1  3:No_2  4:No_3  5:No_4
  6:No_5 7:No_6  8:No_7  9:No_8 10:No_9 11:Point
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os
import re

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

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
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
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if model is None:
        raise HTTPException(status_code=503, detail="OCR model not loaded")

    reading, confidence = _yolo_read(img)

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


def _yolo_read(img: np.ndarray):
    """
    Run YOLO detection, sort boxes left-to-right, assemble the numeric string.
    Returns (reading_str, avg_confidence) or (None, 0.0).
    """
    results = model(img, verbose=False, conf=0.25)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return None, 0.0

    detections = []
    for box in boxes:
        cls_id   = int(box.cls[0].item())
        conf     = float(box.conf[0].item())
        x_center = float(box.xywh[0][0].item())
        char = CLASS_MAP.get(cls_id)
        if char is not None:
            detections.append((x_center, char, conf))

    if not detections:
        return None, 0.0

    # Sort left-to-right
    detections.sort(key=lambda d: d[0])

    reading    = ''.join(d[1] for d in detections)
    avg_conf   = sum(d[2] for d in detections) / len(detections)

    # Sanitise: keep only one decimal point, remove stray chars
    reading = _sanitise(reading)
    if reading is None:
        return None, 0.0

    return reading, avg_conf


def _sanitise(raw: str) -> str | None:
    """Clean up the assembled string into a valid numeric reading."""
    # Allow optional leading minus, digits, one decimal point
    match = re.search(r'-?\d+\.?\d*', raw)
    if not match:
        return None
    return match.group(0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
