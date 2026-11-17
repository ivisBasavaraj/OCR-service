# ===== CELL 1: paste into a Colab code cell and run =====
import cv2, numpy as np

# 7-segment truth table (segments a..g)
PATTERNS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g", " ": "",
}
SEGS = "abcdefg"
KEYS = list(PATTERNS.keys())
PAT = np.array([[1.0 if s in PATTERNS[k] else 0.0 for s in SEGS] for k in KEYS])


# ---------------------------------------------------------------- locating the display
def halo_mask(bgr):
    """Saturated green glow around the LED digits (used only to FIND the display)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g = bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32)
    green = (h >= 55) & (h <= 88) & (s >= 150) & (b < 0.85 * g)
    if not green.any():
        return np.zeros(h.shape, np.uint8)
    vmax = np.percentile(v[green], 99)                    # brightest green in THIS photo
    m = (green & (s >= 0.72 * np.percentile(s[green], 90)) & (v >= 0.80 * vmax)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def find_display(m):
    d = cv2.dilate(m, np.ones((15, 60), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d)
    if n <= 1:
        return None
    best = max(range(1, n), key=lambda j: int(m[lab == j].sum()))
    ys, xs = np.where(m * (lab == best))
    if len(xs) < 40:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def core_mask(bgr, box, q, feature="green", pad=6):
    """Bright segment 'cores' inside the display box (feature above its q-th percentile)."""
    x0, y0, x1, y1 = box
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(bgr.shape[1], x1 + pad + 1), min(bgr.shape[0], y1 + pad + 1)
    patch = bgr[Y0:Y1, X0:X1].astype(np.float32)
    f = patch[..., 1] if feature == "green" else patch.mean(2)
    g = cv2.GaussianBlur(f, (0, 0), 1.2)
    return (g >= np.percentile(g, q)).astype(np.uint8), (X0, Y0, X1, Y1)


# ---------------------------------------------------------------- geometry helpers
def deslant(crop):
    """Level the display (small rotation) and undo the italic slant (shear)."""
    h, w = crop.shape
    f = crop.astype(np.float32)
    # 1) rotation: the angle that makes the rows sharpest
    best = (-1, 0.0)
    for ang in np.arange(-8, 8.1, 1.0):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        r = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR)
        sc = float((r.sum(1) ** 2).sum())
        if sc > best[0]:
            best = (sc, ang)
    ang = best[1]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    f = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR)
    # 2) shear: the slant that makes the columns sharpest
    best = None
    for k in np.linspace(-0.15, 0.45, 41):
        off = abs(k) * h / 2 + 2
        W = int(w + abs(k) * h + 4)
        Ms = np.float32([[1, k, -k * h / 2 + off], [0, 1, 0]])
        wm = cv2.warpAffine(f, Ms, (W, h), flags=cv2.INTER_LINEAR)
        sc = float((wm.sum(0) ** 2).sum())
        if best is None or sc > best[0]:
            best = (sc, k, wm > 0.5)
    return best[2].astype(np.uint8), best[1]


class Integral:
    def __init__(self, m):
        self.I = np.pad(m.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        self.h, self.w = m.shape

    def mean(self, x0, y0, x1, y1):
        x0, x1 = int(round(max(0, x0))), int(round(min(self.w, x1)))
        y0, y1 = int(round(max(0, y0))), int(round(min(self.h, y1)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        I = self.I
        return (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / ((x1 - x0) * (y1 - y0))


def seg_rects(xl, xr, y0, y1, t):
    ym = (y0 + y1) / 2
    return {
        "a": (xl + t, y0, xr - t, y0 + t),
        "g": (xl + t, ym - t / 2, xr - t, ym + t / 2),
        "d": (xl + t, y1 - t, xr - t, y1),
        "f": (xl, y0 + t, xl + t, ym - t / 2),
        "b": (xr - t, y0 + t, xr, ym - t / 2),
        "e": (xl, ym + t / 2, xl + t, y1 - t),
        "c": (xr - t, ym + t / 2, xr, y1 - t),
    }


def decode_cells(shape, ink, xr, y0, y1, w, P, t, n):
    digits, err = [], 0.0
    render = np.zeros(shape, np.uint8)
    for i in range(n):
        xr_i = xr - i * P
        rects = seg_rects(xr_i - w, xr_i, y0, y1, t)
        cov = np.array([ink.mean(*rects[s]) for s in SEGS])
        dist = np.abs(PAT - cov).sum(1)
        j = int(np.argmin(dist))
        err += dist[j]
        digits.append(KEYS[j])
        for s in PATTERNS[KEYS[j]]:
            a, b, c, d = rects[s]
            pad = t * 0.5
            render[int(max(0, b - pad)):int(d + pad), int(max(0, a - pad)):int(c + pad)] = 1
    return digits, err / (7 * n), render


# ---------------------------------------------------------------- one analysis pass
def analyse(crop):
    D, k = deslant(crop)
    rows = np.where(D.sum(1) >= 3)[0]
    cols = np.where(D.sum(0) >= 3)[0]
    if len(rows) < 10 or len(cols) < 10:
        return None
    ymin, ymax = int(rows.min()), int(rows.max()) + 1
    H = ymax - ymin
    ink = Integral(D)
    colsum = D[ymin:ymax].sum(0)
    tall = np.where(colsum >= 0.3 * H)[0]
    if len(tall) == 0:
        return None
    xr0, xl_ink = int(tall.max()) + 1, int(cols.min())
    Db = D.astype(bool)
    best = None
    for tf in (0.14, 0.18):
        t = tf * H
        for wf in np.arange(0.38, 0.92, 0.03):
            w = wf * H
            for pf in np.arange(0.10, 0.80, 0.03):
                P = w + pf * H
                n = int(np.ceil(1 + (xr0 - w - xl_ink - 0.5 * t) / P))
                n = max(1, min(n, 8))
                for dx in (-2, 0, 2):
                    xr = xr0 + dx
                    digits, err, render = decode_cells(D.shape, ink, xr, ymin, ymax, w, P, t, n)
                    outside = float((Db & ~render.astype(bool)).sum()) / max(1, D.sum())
                    lead = max(0.0, (xr - (n - 1) * P - w) - xl_ink) / H
                    score = err + 1.2 * outside + 0.5 * lead
                    if best is None or score < best[0]:
                        best = (score, dict(w=w, P=P, t=t, n=n, xr=xr, digits=digits, err=err, outside=outside))
    score, p = best
    digits, err, render = decode_cells(D.shape, ink, p["xr"], ymin, ymax, p["w"], p["P"], p["t"], p["n"])
    left = (Db & ~render.astype(bool)).astype(np.uint8)       # ink no segment explains
    nlab, lab, st, cen = cv2.connectedComponentsWithStats(left)
    t, cands = p["t"], []
    for j in range(1, nlab):
        bw, bh, area = st[j, 2], st[j, 3], st[j, 4]
        cx, cy = cen[j]
        # a decimal point is a small, compact dot in the lower half of the digits ...
        if not (area >= max(6, 0.06 * t * t) and bw <= 2.0 * t and bh <= 2.0 * t and cy >= ymin + 0.5 * H):
            continue
        # ... sitting just right of a digit's right edge, before the next digit starts
        for i in range(p["n"]):
            right_edge = p["xr"] - i * p["P"]
            if right_edge - 0.25 * t <= cx <= right_edge + (p["P"] - p["w"]) + 0.2 * p["w"]:
                cands.append((area, float(cx), i))
    cands.sort(reverse=True)
    dps = [c[1] for c in cands[:1]]                         # a display shows at most one dot
    dp_after = {c[2] for c in cands[:1]}
    s = ""
    for i in reversed(range(p["n"])):
        s += digits[i]
        if i in dp_after:
            s += "."
    return dict(text=s.strip(), score=score, shear=k, H=H, params=p, dps=dps, D=D)


TARGET_H = 80.0


def read_display(bgr, percentiles=(76, 82, 88)):
    box = find_display(halo_mask(bgr))
    if box is None:
        return None, {"error": "no green LED display found"}
    # bring the digits to a standard size (~80 px tall) so every threshold below is scale-free
    scale = TARGET_H / max(10.0, box[3] - box[1] + 1)
    scale = min(max(scale, 0.25), 6.0)
    if abs(scale - 1.0) > 0.08:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interp)
        box = tuple(int(round(c * scale)) for c in box)
    results = []
    for feature in ("green", "gray"):
        for q in percentiles:
            crop, full_box = core_mask(bgr, box, q, feature)
            r = analyse(crop)
            if r is not None:
                r["q"], r["feature"], r["box"], r["scale"] = q, feature, full_box, scale
                results.append(r)
    if not results:
        return None, {"error": "could not decode"}
    # vote across the threshold variants: the reading most passes agree on wins (ties: best fit)
    votes = {}
    for r in results:
        votes.setdefault(r["text"], []).append(r)
    text = max(votes, key=lambda s: (len(votes[s]), -min(x["score"] for x in votes[s])))
    best = min(votes[text], key=lambda x: x["score"])
    best["confidence"] = len(votes[text]) / len(results)
    best["votes"] = {s: len(v) for s, v in votes.items()}
    return text, best


# ---------------------------------------------------------------- debug picture
def debug_image(info):
    """RGB picture of what was understood: ink (white), unexplained ink (red), digit cells (green/orange)."""
    D, p = info["D"], info["params"]
    rows = np.where(D.sum(1) >= 3)[0]
    ymin, ymax = int(rows.min()), int(rows.max()) + 1
    ink = Integral(D)
    _, _, render = decode_cells(D.shape, ink, p["xr"], ymin, ymax, p["w"], p["P"], p["t"], p["n"])
    vis = np.zeros(D.shape + (3,), np.uint8)
    vis[D > 0] = (255, 255, 255)
    vis[(render > 0) & (D == 0)] = (0, 0, 90)
    vis[(D > 0) & (render == 0)] = (255, 0, 0)
    for i in range(p["n"]):
        xr_i = int(p["xr"] - i * p["P"])
        cv2.line(vis, (xr_i, 0), (xr_i, D.shape[0] - 1), (0, 255, 0), 1)
        cv2.line(vis, (int(xr_i - p["w"]), 0), (int(xr_i - p["w"]), D.shape[0] - 1), (255, 160, 0), 1)
    return cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


# ===== CELL 2: paste into the next cell and run =====
# ===== Upload photos and read the display =====
from google.colab import files
import matplotlib.pyplot as plt

uploaded = files.upload()          # pick one or more photos of the scale
summary = []
for name, data in uploaded.items():
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"{name}: not a readable image"); continue
    text, info = read_display(bgr)
    if text is None:
        print(f"{name}: no green LED display found ({info.get('error')}). Try a closer, sharper photo.")
        plt.figure(figsize=(4, 6)); plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title(name); plt.show()
        continue
    conf = info["confidence"]
    s = info["scale"]
    x0, y0, x1, y1 = [int(c / s) for c in info["box"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].imshow(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)); ax[0].axis("off")
    ax[0].set_title(f"{name}  ->  {text}", fontsize=13)
    ax[1].imshow(debug_image(info)); ax[1].axis("off")
    ax[1].set_title(f"what was understood (confidence {conf*100:.0f}%)", fontsize=10)
    plt.tight_layout(); plt.show()
    flag = "" if conf >= 0.6 else "   <-- UNSURE: compare with the picture above"
    print(f"{name}: {text}   (confidence {conf*100:.0f}%){flag}")
    summary.append((name, text, conf))

print("\nSUMMARY")
for name, text, conf in summary:
    print(f"  {text:>9}   {conf*100:3.0f}%   {name}")
