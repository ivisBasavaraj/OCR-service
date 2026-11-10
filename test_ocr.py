"""
Test OCR detection against all test images with ground truth comparison.
Usage: python test_ocr.py [base_url]
"""
import sys
import requests
from pathlib import Path

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://ocr.iqonixlabs.io"

# class_id → digit char (matches data.yaml)
CLASS_MAP = {
    0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
    6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: '.',
}

DATASET_ROOT = Path(__file__).parent.parent / "Digital weight scale detection 2.v1i.yolo26"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
TEST_DIR = DATASET_ROOT / SPLIT
IMG_DIR   = TEST_DIR / "images"
LBL_DIR   = TEST_DIR / "labels"


def parse_ground_truth(label_path: Path) -> str:
    """Parse YOLO label file → reconstruct reading string sorted by x_center."""
    detections = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls_id   = int(parts[0])
        x_center = float(parts[1])
        char = CLASS_MAP.get(cls_id, '?')
        detections.append((x_center, char))
    detections.sort(key=lambda d: d[0])
    return ''.join(d[1] for d in detections)


def run_tests():
    print(f"Testing against: {BASE_URL}\n")

    # Health check
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Health: {r.json()}\n")

    images = sorted(IMG_DIR.glob("*.jpg"))
    passed = 0
    total  = len(images)

    for img_path in images:
        label_path = LBL_DIR / img_path.with_suffix(".txt").name
        gt = parse_ground_truth(label_path) if label_path.exists() else "N/A"

        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/read-scale",
                files={"file": (img_path.name, f, "image/jpeg")},
                timeout=30,
            )

        if resp.status_code == 200:
            data = resp.json()
            predicted = data["reading"]
            conf      = data["confidence"]
            # Exact match on the numeric value
            ok = predicted == gt or (
                _safe_float(predicted) is not None and
                _safe_float(gt) is not None and
                abs(_safe_float(predicted) - _safe_float(gt)) < 0.01
            )
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"{status} | GT: {gt:>10} | Pred: {predicted:>10} | Conf: {conf:.3f} | {img_path.name[:30]}")
        else:
            print(f"FAIL | GT: {gt:>10} | Error: {resp.json().get('detail')} | {img_path.name[:30]}")

    print(f"\nResult: {passed}/{total} = {passed/total*100:.1f}% accuracy")


def _safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    run_tests()
