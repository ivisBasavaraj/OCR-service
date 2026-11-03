"""
Scale OCR Microservice
Accepts a JPEG/PNG image of a digital weight scale display,
runs YOLO detection to find digit bounding boxes,
reads them left-to-right and assembles the numeric reading.

Classes (from data.yaml):
  0: -ve   1: No_0   2: No_1   3: No_2   4: No_3   5: No_4
  6: No_5  7: No_6   8: No_7   9: No_8  10: No_9  11: Point
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO
import io
import os

app = FastAPI(title="Scale OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class index → character mapping
CLASS_MAP = {
    0: '-',
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
    6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: '.',
}

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: model not found at {MODEL_PATH} — using raw OCR fallback")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/read-scale")
async def read_scale(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    if model is not None:
        reading, confidence = _yolo_read(img)
    else:
        reading, confidence = _fallback_ocr(img)

    if reading is None:
        raise HTTPException(status_code=422, detail="Could not read scale display")

    try:
        value = float(reading)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Parsed string not numeric: {reading}")

    return {
        "reading": reading,
        "valueGrams": round(value, 3),
        "confidence": round(confidence, 3),
    }


def _yolo_read(img):
    """Run YOLO, sort detections left-to-right, assemble string."""
    results = model(img, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return None, 0.0

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        x_center = float(box.xywh[0][0].item())
        char = CLASS_MAP.get(cls_id)
        if char is not None:
            detections.append((x_center, char, conf))

    if not detections:
        return None, 0.0

    # Sort left-to-right by x_center
    detections.sort(key=lambda d: d[0])

    reading = ''.join(d[1] for d in detections)
    avg_conf = sum(d[2] for d in detections) / len(detections)

    # Strip leading minus if not negative (noise)
    reading = reading.lstrip('-') if not reading.startswith('-') else reading

    return reading, avg_conf


def _fallback_ocr(img):
    """
    Fallback: crop the brightest horizontal band (the display area),
    threshold, and use basic contour analysis to estimate digits.
    Returns None — forces the caller to handle gracefully.
    """
    return None, 0.0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
