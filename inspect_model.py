import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO('best.pt')
img_dir = Path(r'D:\Pushpa Chains\Digital weight scale detection 2.v1i.yolo26\valid\images')
lbl_dir = Path(r'D:\Pushpa Chains\Digital weight scale detection 2.v1i.yolo26\valid\labels')

CLASS_MAP = {0:'-',1:'0',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6',8:'7',9:'8',10:'9',11:'.'}

for img_path in sorted(img_dir.glob('*.jpg'))[:5]:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Ground truth
    lbl = lbl_dir / img_path.with_suffix('.txt').name
    gt_dets = []
    for line in lbl.read_text().strip().splitlines():
        p = line.split()
        gt_dets.append((float(p[1]) * w, CLASS_MAP.get(int(p[0]),'?'), float(p[3])*w, float(p[4])*h))
    gt_dets.sort(key=lambda x: x[0])
    gt_str = ''.join(d[1] for d in gt_dets)
    gt_box_sizes = [(round(d[2],1), round(d[3],1)) for d in gt_dets]

    print(f'\n{img_path.name}  img={w}x{h}')
    print(f'  GT: {gt_str}  box_sizes(px): {gt_box_sizes}')

    for thresh in [0.30, 0.15, 0.05]:
        results = model(img, verbose=False, conf=thresh, imgsz=640)[0]
        boxes = results.boxes
        n = len(boxes) if boxes else 0
        if n:
            dets = []
            for b in boxes:
                cx = float(b.xywh[0][0])
                bw = float(b.xywh[0][2])
                bh = float(b.xywh[0][3])
                ch = CLASS_MAP.get(int(b.cls[0]), '?')
                cf = float(b.conf[0])
                dets.append((cx, ch, cf, round(bw,1), round(bh,1)))
            dets.sort(key=lambda x: x[0])
            pred_str = ''.join(d[1] for d in dets)
            print(f'  PRED(conf>={thresh}): {pred_str}  boxes: {[(d[1],round(d[2],2),d[3],d[4]) for d in dets]}')
            break
        else:
            print(f'  PRED(conf>={thresh}): 0 boxes')
