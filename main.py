"""FastAPI adapter for the model-free seven-segment scale reader.

The recognition implementation lives in ``scale_reader_cells.py`` and is the
supplied Colab algorithm. This module intentionally has no trained-model or
GPU dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


def _load_reader() -> dict[str, Any]:
    """Load only cell 1; cell 2 is Colab's interactive upload UI."""
    source = Path(__file__).with_name("scale_reader_cells.py").read_text(encoding="utf-8")
    algorithm, _, _ = source.partition("# ===== CELL 2:")
    namespace: dict[str, Any] = {}
    exec(compile(algorithm, "scale_reader_cells.py", "exec"), namespace)
    return namespace


_reader = _load_reader()
read_display = _reader["read_display"]

app = FastAPI(title="Green LED Scale Reader", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "reader": "seven-segment", "model_loaded": False}


@app.post("/read-scale")
async def read_scale(file: UploadFile = File(...)) -> dict[str, object]:
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    reading, info = read_display(image)
    if reading is None:
        raise HTTPException(status_code=422, detail=info.get("error", "Could not decode scale display"))

    try:
        value = float(reading)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Non-numeric reading: '{reading}'") from error

    return {
        "reading": reading,
        "valueGrams": round(value, 3),
        "confidence": round(float(info["confidence"]), 3),
        "isNegative": value < 0,
    }
