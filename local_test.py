"""
Local pipeline test — runs main.py logic directly on validation images.
Usage: python local_test.py
"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from pathlib import Path

# Patch env before import
import os
os.environ['MODEL_PATH'] = str(Path(__file__).parent / 'best.pt')

from main import _read_scale, load_model, model as _m
import main

# Load model
main.load_model()

CLASS_MAP = {0:'-',1:'0',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6',8:'7',9:'8',10:'9',11:'.'}

DATASET = Path(r'D:\Pushpa Chains\Digital weight scale detection 2.v1i.yolo26')

def parse_gt(lbl_path, img_w, img_h):
    dets = []
    for line in lbl_path.read_text().strip().splitlines():
        p = line.split()
        dets.append((float(p[1]) * img_w, CLASS_MAP.get(int(p[0]), '?')))
    dets.sort(key=lambda x: x[0])
    return ''.join(d[1] for d in dets)

def safe_float(s):
    try: return float(s)
    except: return None

for split in ['valid', 'test']:
    img_dir = DATASET / split / 'images'
    lbl_dir = DATASET / split / 'labels'
    images  = sorted(img_dir.glob('*.jpg'))
    passed  = 0

    print(f'\n=== {split.upper()} ({len(images)} images) ===')
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lbl  = lbl_dir / img_path.with_suffix('.txt').name
        gt   = parse_gt(lbl, w, h)

        reading, conf = _read_scale(img)

        ok = False
        if reading and gt:
            gf, pf = safe_float(gt), safe_float(reading)
            ok = (gf is not None and pf is not None and abs(gf - pf) < 0.01)

        status = 'PASS' if ok else 'FAIL'
        if ok: passed += 1
        print(f'  {status} | GT:{gt:>10} | Pred:{str(reading):>10} | conf={conf:.3f} | {img_path.name[:35]}')

    print(f'  --> {passed}/{len(images)} = {passed/len(images)*100:.1f}%')
