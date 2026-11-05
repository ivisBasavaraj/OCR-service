"""
Train YOLOv8n on the digital weight scale dataset.
Run: python train.py
Output: best.pt in this directory
"""

from ultralytics import YOLO
import shutil, os

DATA_YAML = r"D:\Pushpa Chains\Digital weight scale detection 2.v1i.yolo26\data.yaml"
EPOCHS    = 150
IMG_SIZE  = 640
BATCH     = 4       # small dataset — small batch
PROJECT   = "runs/detect"
NAME      = "scale_ocr_v2"

def train():
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=40,
        device="cpu",
        workers=2,
        exist_ok=True,
        # ── Augmentation — safe for digits ──────────────────────────
        fliplr=0.0,       # NO horizontal flip (6 ≠ 9, 2 ≠ 5)
        flipud=0.0,       # NO vertical flip
        degrees=5.0,      # slight rotation only
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0001,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=0.5,
        mixup=0.0,
        # ── Optimiser ───────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        label_smoothing=0.1,
    )

    best = os.path.join(PROJECT, NAME, "weights", "best.pt")
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    shutil.copy(best, dest)
    print(f"\n✅  Model saved to {dest}")

if __name__ == "__main__":
    train()
