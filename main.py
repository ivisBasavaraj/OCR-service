"""
Scale OCR Microservice — FastAPI
Pipeline:
  1. Upscale full image to 1280px wide (digits become ~20px wide)
  2. Detect LCD digit strip (brightest horizontal band)
  3. Crop + upscale strip to 640px tall
  4. Run YOLO with tight dedup (digits are ~5px wide in original)
  5. Assemble reading left-to-right
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
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")


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


# ── Upscale ───────────────────────────────────────────────────────────────────

def _upscale(img: np.ndarray, target_w: int = 1280) -> np.ndarray:
    """Upscale image so digits go from ~5px to ~20px wide."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    scale = target_w / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_CUBIC)


# ── LCD strip detection ───────────────────────────────────────────────────────

def _find_digit_strip(img: np.ndarray):
    """
    Find the horizontal band containing the LCD digits.
    Returns cropped strip or original image if detection fails.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 31), 0)  # blur only vertically

    row_mean = np.mean(blurred, axis=1)
    smooth = np.convolve(row_mean, np.ones(10) / 10, mode='same')

    peak = int(np.argmax(smooth))
    threshold = smooth[peak] * 0.70

    top = peak
    while top > 0 and smooth[top] > threshold:
        top -= 1

    bottom = peak
    while bottom < h - 1 and smooth[bottom] > threshold:
        bottom += 1

    pad = max(5, int((bottom - top) * 0.3))
    top    = max(0, top - pad)
    bottom = min(h, bottom + pad)

    strip_h = bottom - top

    # Only use strip if it's a meaningful sub-region (5%–60% of image height)
    if h * 0.05 < strip_h < h * 0.60:
        strip = img[top:bottom, :]
        # Upscale strip height to at least 128px so digits are tall enough
        if strip_h < 128:
            scale = 128 / strip_h
            new_w = int(w * scale)
            strip = cv2.resize(strip, (new_w, 128), interpolation=cv2.INTER_CUBIC)
        return strip

    return img


# ── Preprocessing variants ────────────────────────────────────────────────────

def _variants(img: np.ndarray):
    out = [("original", img)]

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    clahe = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    out.append(("clahe", clahe))

    # Sharpen
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    out.append(("clahe+sharp", cv2.filter2D(clahe, -1, k)))

    # Brightness boost
    out.append(("bright", cv2.convertScaleAbs(img, alpha=1.5, beta=50)))

    return out


# ── YOLO inference ────────────────────────────────────────────────────────────

def _yolo_read(img: np.ndarray):
    """Run YOLO, deduplicate tightly, sort left-to-right."""
    for conf_thresh in [0.25, 0.15, 0.08]:
        results = model(img, verbose=False, conf=conf_thresh, imgsz=640)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            continue

        dets = []
        for box in boxes:
            cls_id   = int(box.cls[0].item())
            conf     = float(box.conf[0].item())
            x_center = float(box.xywh[0][0].item())
            box_w    = float(box.xywh[0][2].item())
            char = CLASS_MAP.get(cls_id)
            if char is not None:
                dets.append((x_center, char, conf, box_w))

        if not dets:
            continue

        # Tight dedup: overlap = 40% of median box width
        median_w = float(np.median([d[3] for d in dets]))
        overlap  = max(2.0, median_w * 0.4)
        dets = _deduplicate(dets, overlap)
        dets.sort(key=lambda d: d[0])

        reading  = ''.join(d[1] for d in dets)
        avg_conf = sum(d[2] for d in dets) / len(dets)

        reading = _sanitise(reading)
        if reading is None:
            continue

        return reading, avg_conf, len(dets)

    return None, 0.0, 0


def _deduplicate(dets, overlap_px):
    dets = sorted(dets, key=lambda d: d[2], reverse=True)
    kept = []
    for det in dets:
        if not any(abs(det[0] - k[0]) < overlap_px for k in kept):
            kept.append(det)
    return kept


# ── Full pipeline ─────────────────────────────────────────────────────────────

def _read_scale(img: np.ndarray):
    # Step 1: upscale so digits are bigger
    img_up = _upscale(img, target_w=1280)

    # Step 2: find digit strip
    strip = _find_digit_strip(img_up)

    # Step 3: try all preprocessing variants on strip, fallback to full upscaled
    sources = [strip]
    if strip is not img_up:
        sources.append(img_up)

    best_reading = None
    best_conf    = 0.0
    best_score   = -1

    for src in sources:
        for name, processed in _variants(src):
            reading, conf, n = _yolo_read(processed)
            if reading is None:
                continue
            score = n * conf * (1.3 if '.' in reading else 1.0)
            print(f"  [{name}] {reading}  conf={conf:.3f}  n={n}  score={score:.3f}")
            if score > best_score:
                best_score   = score
                best_reading = reading
                best_conf    = conf

    return best_reading, best_conf


# ── Sanitise + decimal inference ─────────────────────────────────────────────

def _infer_decimal(digits: str) -> str:
    """
    Gold scales always show 2 decimal places.
    Insert decimal point based on digit count:
      3 digits -> X.XX   (003 -> 0.03)
      4 digits -> XX.XX  (8837 -> 88.37)
      5 digits -> XXX.XX (56708 -> 567.08)
      6 digits -> XXXX.XX
    """
    neg = digits.startswith('-')
    d = digits.lstrip('-')
    n = len(d)
    if n == 3:
        result = d[0] + '.' + d[1:]
    elif n == 4:
        result = d[:2] + '.' + d[2:]
    elif n == 5:
        result = d[:3] + '.' + d[3:]
    elif n == 6:
        result = d[:4] + '.' + d[4:]
    else:
        result = d
    return ('-' + result) if neg else result


def _sanitise(raw: str) -> str | None:
    match = re.search(r'-?\d+\.?\d*', raw)
    if not match:
        return None
    val_str = match.group(0)
    # If no decimal detected, infer it (dot is ~1px wide, often missed)
    if '.' not in val_str:
        val_str = _infer_decimal(val_str)
    try:
        val = float(val_str)
        if val < 0.0 or val > 9999.999:
            return None
    except ValueError:
        return None
    return val_str


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
