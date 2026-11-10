"""
Scale OCR Microservice — FastAPI
Pipeline:
  1. Detect the LCD display region (brightest horizontal band)
  2. Crop + upscale that region to 640px height
  3. Run YOLO digit detection on the upscaled crop
  4. Assemble reading left-to-right
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

# class index → character  (matches data.yaml order)
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

    reading, confidence = _read_scale(img)

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


# ── Step 1: LCD region detection ─────────────────────────────────────────────

def _find_lcd_roi(img: np.ndarray):
    """
    Find the LCD display region using brightness analysis.
    LCD displays are typically the brightest horizontal band in the image.
    Returns (x1, y1, x2, y2) or None to use full image.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Find brightest row band (LCD backlight)
    row_brightness = np.mean(blurred, axis=1)
    smooth = np.convolve(row_brightness, np.ones(20) / 20, mode='same')

    # Find peak brightness region
    peak_row = int(np.argmax(smooth))
    threshold = smooth[peak_row] * 0.75

    # Expand from peak until brightness drops below threshold
    top = peak_row
    while top > 0 and smooth[top] > threshold:
        top -= 1

    bottom = peak_row
    while bottom < h - 1 and smooth[bottom] > threshold:
        bottom += 1

    # Add 20% padding
    pad_v = max(10, int((bottom - top) * 0.2))
    top = max(0, top - pad_v)
    bottom = min(h, bottom + pad_v)

    # For columns: use center 80% (avoid scale frame edges)
    left = int(w * 0.05)
    right = int(w * 0.95)

    roi_h = bottom - top
    roi_w = right - left

    # Only use ROI if it's a meaningful crop (not the whole image)
    if roi_h < h * 0.8 and roi_h > h * 0.03:
        return left, top, right, bottom

    return None


def _crop_and_upscale(img: np.ndarray, roi):
    """Crop to ROI and upscale so digit height is ~40-60px (good for YOLO)."""
    if roi is None:
        return img

    x1, y1, x2, y2 = roi
    crop = img[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return img

    # Upscale so the crop height is at least 320px
    target_h = max(320, h)
    if h < target_h:
        scale = target_h / h
        new_w = int(w * scale)
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    return crop


# ── Step 2: Preprocessing variants ───────────────────────────────────────────

def _preprocess_variants(img: np.ndarray):
    variants = [("original", img.copy())]

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    variants.append(("clahe", clahe_img))

    # Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants.append(("clahe+sharp", cv2.filter2D(clahe_img, -1, kernel)))

    # Brightness boost (dark displays)
    variants.append(("bright", cv2.convertScaleAbs(img, alpha=1.4, beta=40)))

    # Gamma correction (overexposed displays)
    gamma_img = np.power(img / 255.0, 1.5) * 255
    variants.append(("gamma", gamma_img.astype(np.uint8)))

    return variants


# ── Step 3: YOLO inference ────────────────────────────────────────────────────

def _yolo_read(img: np.ndarray):
    """
    Run YOLO, deduplicate, sort left-to-right, return (reading, conf, n_digits).
    """
    h, w = img.shape[:2]

    for conf_thresh in [0.30, 0.20, 0.12]:
        results = model(img, verbose=False, conf=conf_thresh, imgsz=640)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            continue

        detections = []
        for box in boxes:
            cls_id   = int(box.cls[0].item())
            conf     = float(box.conf[0].item())
            x_center = float(box.xywh[0][0].item())
            box_w    = float(box.xywh[0][2].item())
            char = CLASS_MAP.get(cls_id)
            if char is not None:
                detections.append((x_center, char, conf, box_w))

        if not detections:
            continue

        # Adaptive dedup: overlap threshold = median box width * 0.5
        widths = [d[3] for d in detections]
        median_w = float(np.median(widths)) if widths else 15
        overlap_thresh = max(8, median_w * 0.5)

        detections = _deduplicate(detections, overlap_thresh)
        detections.sort(key=lambda d: d[0])

        reading  = ''.join(d[1] for d in detections)
        avg_conf = sum(d[2] for d in detections) / len(detections)

        reading = _sanitise(reading)
        if reading is None:
            continue

        return reading, avg_conf, len(detections)

    return None, 0.0, 0


def _deduplicate(detections, overlap_px):
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    for det in detections:
        x = det[0]
        if not any(abs(x - k[0]) < overlap_px for k in kept):
            kept.append(det)
    return kept


# ── Step 4: Best result selection ────────────────────────────────────────────

def _read_scale(img: np.ndarray):
    """Full pipeline: ROI crop → upscale → multi-variant inference → best pick."""
    roi = _find_lcd_roi(img)
    cropped = _crop_and_upscale(img, roi)

    # Also try full image as fallback
    candidates = [cropped]
    if roi is not None:
        candidates.append(img)  # fallback: full image

    best_reading = None
    best_conf = 0.0
    best_score = -1

    for source_img in candidates:
        for name, processed in _preprocess_variants(source_img):
            reading, conf, n_digits = _yolo_read(processed)
            if reading is None:
                continue

            has_dot = '.' in reading
            score = n_digits * conf * (1.3 if has_dot else 1.0)

            print(f"  [{name}] reading={reading} conf={conf:.3f} digits={n_digits} score={score:.3f}")

            if score > best_score:
                best_score = score
                best_reading = reading
                best_conf = conf

    return best_reading, best_conf


# ── Sanitise ──────────────────────────────────────────────────────────────────

def _sanitise(raw: str) -> str | None:
    match = re.search(r'-?\d+\.?\d*', raw)
    if not match:
        return None
    val_str = match.group(0)
    try:
        val = float(val_str)
        # Valid range: 0.000g (zero/tare) to 9999g
        if val < 0.0 or val > 9999.999:
            return None
    except ValueError:
        return None
    return val_str


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
