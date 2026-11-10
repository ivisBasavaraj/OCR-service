"""
Train YOLOv8s on the digital weight scale dataset at 1280px.
Matches inference resolution so digits are ~20px wide during both train and test.
Run: python train.py
"""

from ultralytics import YOLO
import shutil, os

DATA_YAML = r"D:\Pushpa Chains\Digital weight scale detection 2.v1i.yolo26\data.yaml"
EPOCHS    = 300
IMG_SIZE  = 1280   # upscaled resolution — digits go from 5px to 20px wide
BATCH     = 2      # 1280px needs more memory, keep batch small on CPU
PROJECT   = "runs/detect"
NAME      = "scale_ocr_v3"


def train():
    # Use YOLOv8s (small) — more capacity than nano for tiny digit classification
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=60,
        device="cpu",
        workers=2,
        exist_ok=True,

        # ── Augmentation ────────────────────────────────────────────
        fliplr=0.0,        # NO horizontal flip (6 != 9)
        flipud=0.0,        # NO vertical flip
        degrees=3.0,       # minimal rotation
        translate=0.05,
        scale=0.2,
        shear=1.0,
        perspective=0.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,         # strong brightness variation (LCD varies a lot)
        mosaic=0.0,        # OFF — mosaic breaks digit spatial layout
        mixup=0.0,
        erasing=0.0,       # OFF — erasing removes digits

        # ── Optimiser ────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        label_smoothing=0.05,

        # ── Loss weights — boost cls so digit identity is learned better
        box=5.0,
        cls=1.5,
        dfl=1.5,
    )

    best = os.path.join(PROJECT, NAME, "weights", "best.pt")
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    shutil.copy(best, dest)
    print(f"\nModel saved to {dest}")


if __name__ == "__main__":
    train()
