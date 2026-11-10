"""
Scale OCR Microservice — FastAPI
Pipeline: YOLOv8 (LCD region detection) + CRNN (digit recognition)
Based on: https://github.com/OICWS/lcd-digit-recognition
"""

import os
import io
import json
import re
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Scale OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR   = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))
YOLO_PATH   = MODEL_DIR / "best.pt"
CRNN_PATH   = MODEL_DIR / "final_crnn.pth"
CONFIG_PATH = MODEL_DIR / "crnn_config.json"

_yolo  = None
_crnn  = None
_cfg   = None

# ── CRNN Architecture (must match training) ───────────────────────────────────
class CRNN(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.3)
        self.fc  = nn.Linear(512, num_class)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        o, _ = self.rnn(f)
        return self.fc(o).permute(1, 0, 2)


TRANSFORM = T.Compose([
    T.Resize((64, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


# ── Startup: load models ──────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global _yolo, _crnn, _cfg

    if not YOLO_PATH.exists():
        print(f"⚠️  YOLO weights not found at {YOLO_PATH}")
        return
    if not CRNN_PATH.exists():
        print(f"⚠️  CRNN weights not found at {CRNN_PATH}")
        return
    if not CONFIG_PATH.exists():
        print(f"⚠️  CRNN config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH) as f:
        _cfg = json.load(f)
    _cfg.setdefault("min_val", 0.1)
    _cfg.setdefault("max_val", 9999.9)

    from ultralytics import YOLO
    _yolo = YOLO(str(YOLO_PATH))
    print("✅ YOLO loaded")

    _crnn = CRNN(_cfg["num_class"])
    _crnn.load_state_dict(torch.load(str(CRNN_PATH), map_location="cpu"))
    _crnn.eval()
    print("✅ CRNN loaded")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_loaded": _yolo is not None,
        "crnn_loaded": _crnn is not None,
    }


# ── Inference endpoint ────────────────────────────────────────────────────────
@app.post("/read-scale")
async def read_scale(file: UploadFile = File(...)):
    if _yolo is None or _crnn is None:
        raise HTTPException(status_code=503, detail="OCR models not loaded")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    value, confidence = _run_pipeline(img_cv)

    if value is None:
        raise HTTPException(
            status_code=422,
            detail="Could not detect or read digits on scale display",
        )

    return {
        "reading": str(value),
        "valueGrams": round(value, 3),
        "confidence": round(confidence, 3),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────
def _run_pipeline(img_cv: np.ndarray):
    """YOLOv8 detects LCD region → CRNN reads digits."""

    # Step 1: YOLO — find LCD bounding box
    results = _yolo(img_cv, verbose=False, conf=0.3)
    if not results or len(results[0].boxes) == 0:
        print("  [YOLO] No LCD region detected")
        return None, 0.0

    boxes    = results[0].boxes
    best_idx = boxes.conf.argmax().item()
    box      = boxes.xyxy[best_idx].cpu().numpy()
    yolo_conf = boxes.conf[best_idx].item()
    print(f"  [YOLO] conf={yolo_conf:.2f}")

    # Step 2: Crop LCD region
    crop = _crop_lcd(img_cv, box)
    if crop is None or crop.size == 0:
        return None, 0.0

    # Step 3: CRNN — read digits from crop
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_pil = _smart_resize(crop_pil)
    tensor   = TRANSFORM(crop_pil).unsqueeze(0)

    with torch.no_grad():
        log_probs = _crnn(tensor).log_softmax(2)

    pred_str, crnn_conf = _decode(log_probs[:, 0, :])
    print(f"  [CRNN] pred='{pred_str}' conf={crnn_conf:.3f}")

    value = _validate(pred_str)
    return value, crnn_conf


def _crop_lcd(img_cv, box_xyxy, pad=5):
    h, w = img_cv.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img_cv[y1:y2, x1:x2]


def _smart_resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w > 400:
        scale = 300 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _decode(log_probs_t):
    """CTC greedy decode with confidence."""
    chars = _cfg["chars"]
    blank = _cfg["blank"]
    probs = log_probs_t.exp()
    ids   = log_probs_t.argmax(-1).tolist()
    max_p = probs.max(-1).values.tolist()

    out_chars, char_confs, prev = [], [], None
    for i, p in zip(ids, max_p):
        if i != blank and i != prev:
            out_chars.append(chars[i])
            char_confs.append(p)
        prev = i

    pred = "".join(out_chars)
    conf = float(min(char_confs)) if char_confs else 0.0
    return pred, conf


def _validate(pred_str: str):
    """Validate format and range."""
    if not re.match(r"^\d+\.?\d*$", pred_str):
        print(f"  [WARN] Invalid format: '{pred_str}'")
        return None
    try:
        val = float(pred_str)
    except ValueError:
        return None
    min_val = _cfg.get("min_val", 0.1)
    max_val = _cfg.get("max_val", 9999.9)
    if not (min_val <= val <= max_val):
        print(f"  [WARN] Out of range [{min_val}, {max_val}]: {val}")
        return None
    return val


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
