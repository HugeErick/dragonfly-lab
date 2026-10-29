from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 2400  # detection / editing proxy size; exports are full resolution

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT if (_ROOT / "pyproject.toml").exists() else Path.cwd()
# Annotation output goes to  <project>/src/<image name>/
OUTPUT_ROOT = PROJECT_ROOT / "src"


# --------------------------------------------------------------------------- IO
def imread(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise IOError(f"Cannot encode {path}")
    buf.tofile(str(path))


def make_proxy(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def clean_stem(src: Path) -> str:
  name = src.name
  while Path(name).suffix.lower() in _IMG_EXTS:
    name = Path(name).stem
  return name


def output_dir_for(src: Path) -> Path:
    return OUTPUT_ROOT / clean_stem(src)


def load_mask(out_dir: Path, shape: tuple[int, int], stem: str) -> np.ndarray | None:
    p = out_dir / f"{stem}_mask.png"
    if not p.exists():
        return None
    m = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return ((m > 127) * 255).astype(np.uint8)


# ----------------------------------------------------------- body & wing finder
@dataclass
class BodyWings:
    body: np.ndarray            # uint8 0/255
    wings: np.ndarray           # uint8 label image, 0 = none, 1..N wings
    body_contours: list
    wing_contours: list         # wing_contours[i] belongs to label i+1
    wing_centroids: list


def _odd(n: float) -> int:
    n = int(n)
    return n if n % 2 else n + 1


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_external(mask: np.ndarray, min_area: float = 0) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    keep = [c for c in cnts if cv2.contourArea(c) >= min_area]
    cv2.drawContours(out, keep, -1, 255, cv2.FILLED)
    return out


def _background_distance(lab: np.ndarray) -> tuple[np.ndarray, float]:
    """Distance (Lab) of each pixel to the colour clusters found on the image border."""
    h, w = lab.shape[:2]
    b = max(4, int(min(h, w) * 0.03))
    border = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[b:-b, :b].reshape(-1, 3), lab[b:-b, -b:].reshape(-1, 3),
    ]).astype(np.float32)
    if len(border) > 20000:
        border = border[np.random.default_rng(0).choice(len(border), 20000, replace=False)]
    k = 3
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(border, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    keep = centers[counts >= 0.08 * len(border)]  # ignore tiny clusters (wing tips touching the border)
    if len(keep) == 0:
        keep = centers[[counts.argmax()]]
    dist = np.min([np.linalg.norm(lab - c, axis=2) for c in keep], axis=0).astype(np.float32)
    edge = np.concatenate([dist[:b].ravel(), dist[-b:].ravel(), dist[:, :b].ravel(), dist[:, -b:].ravel()])
    return dist, float(np.percentile(edge, 75))


def detect_body_and_wings(bgr: np.ndarray, sensitivity: float = 1.0) -> BodyWings:
    h, w = bgr.shape[:2]
    m = min(h, w)
    s = max(0.3, sensitivity)

    blur = cv2.GaussianBlur(bgr, (0, 0), 1.5)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist, floor = _background_distance(lab)

    lo = max(5.0, floor * 1.5 + 2.0) / s       # anything noticeably different from the backdrop
    hi = max(lo * 3.0, 25.0) / s               # strongly different (dark, opaque body)

    # --- foreground: colour difference  +  dense vein texture (translucent wings)
    weak = (dist > lo).astype(np.uint8) * 255
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    edges = cv2.Canny(gray, 10 / s, 30 / s)
    k = _odd(max(9, m * 0.02))
    density = cv2.blur(edges.astype(np.float32) / 255.0, (k, k))
    texture = (density > 0.05).astype(np.uint8) * 255
    fg = cv2.bitwise_or(weak, texture)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _ellipse(3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _ellipse(_odd(max(5, m * 0.01))))
    min_area = 0.002 * h * w
    fg = _fill_external(fg, min_area)

    # --- body: thick, strongly different structure that survives a big opening
    strong = (dist > hi).astype(np.uint8) * 255
    r = _odd(max(5, m * 0.008))
    core = cv2.morphologyEx(strong, cv2.MORPH_OPEN, _ellipse(r))
    body = np.zeros_like(core)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        core = ((lbl == big) * 255).astype(np.uint8)
        grown = cv2.dilate(core, _ellipse(2 * r + 1))      # recover the thin abdomen tip etc.
        body = cv2.bitwise_and(grown, cv2.morphologyEx(strong, cv2.MORPH_CLOSE, _ellipse(3)))
        body = cv2.bitwise_or(body, core)
        body = _fill_external(cv2.morphologyEx(body, cv2.MORPH_CLOSE, _ellipse(5)))

    # --- wings: foreground minus body, split into connected pieces
    wing_area = cv2.bitwise_and(fg, cv2.bitwise_not(cv2.dilate(body, _ellipse(5))))
    wing_area = cv2.morphologyEx(wing_area, cv2.MORPH_OPEN, _ellipse(5))
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(wing_area, connectivity=8)
    order = [i for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]) if i != 0
             and stats[i, cv2.CC_STAT_AREA] >= min_area][:8]
    wings = np.zeros((h, w), np.uint8)
    wing_cnts, wing_cent = [], []
    for new_id, i in enumerate(order, start=1):
        comp = (lbl == i).astype(np.uint8)
        wings[comp > 0] = new_id
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        wing_cnts.append(cv2.approxPolyDP(c, 1.5, True))
        wing_cent.append((float(cents[i][0]), float(cents[i][1])))

    bc, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_cnts = [cv2.approxPolyDP(c, 1.5, True) for c in bc if cv2.contourArea(c) > 20]
    return BodyWings(body, wings, body_cnts, wing_cnts, wing_cent)


def exclusion_zone(det: BodyWings | None, margin: float, shape: tuple[int, int]) -> np.ndarray:
    """Whole dragonfly (body + wings, gaps between them bridged) grown by `margin` px. uint8 0/255."""
    h, w = shape
    zone = np.zeros((h, w), np.uint8)
    if det is None:
        return zone
    fly = (((det.body > 0) | (det.wings > 0)).astype(np.uint8)) * 255
    if not fly.any():
        return zone
    fly = cv2.morphologyEx(fly, cv2.MORPH_CLOSE, _ellipse(_odd(min(61, max(5, margin)))))
    fly = _fill_external(fly)                           # fill holes between wings / body
    if margin > 0:
        d = cv2.distanceTransform(255 - fly, cv2.DIST_L2, 5)
        zone[d <= margin] = 255
    zone[fly > 0] = 255
    return zone


# ------------------------------------------------------------------ dust finder
@dataclass
class Speck:
    cx: float
    cy: float
    r: float
    contrast: float
    dark: bool
    contour: np.ndarray


def detect_dust(bgr: np.ndarray, exclude: np.ndarray | None = None, sensitivity: float = 5.0,
                max_diam: int = 20) -> list[Speck]:
    """Tiny compact blobs darker/brighter than their surroundings.

    Any blob that touches the `exclude` zone (dragonfly + margin) is discarded entirely.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = _odd(max(7, max_diam * 2.5))
    local = cv2.medianBlur(gray, k)                       # speck-free local background
    diff = gray.astype(np.int16) - local.astype(np.int16)
    ad = np.abs(diff)
    sigma = max(1.0, 1.4826 * float(np.median(ad)))
    thr = max(4.0 + (10.0 - sensitivity), (12.0 - sensitivity) * sigma)

    mask = (ad > thr).astype(np.uint8)

    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return []
    flat = lbl.ravel()
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    longest, shortest = np.maximum(bw, bh), np.maximum(1, np.minimum(bw, bh))
    ok = (np.arange(n) > 0) & (area >= 3) & (longest <= max_diam) \
        & (area / (bw * bh) >= 0.4) & (longest / shortest <= 3.0)    # compact: rejects veins / edges
    if exclude is not None and exclude.any():
        touches = np.bincount(flat, weights=(exclude.ravel() > 0).astype(np.float32), minlength=n) > 0
        ok &= ~touches
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []

    contrast = np.bincount(flat, weights=ad.ravel(), minlength=n) / np.maximum(area, 1)
    signed = np.bincount(flat, weights=diff.ravel(), minlength=n)
    idx = idx[np.argsort(-contrast[idx])][:3000]

    specks: list[Speck] = []
    for i in idx:
        x, y, ww, hh = stats[i, :4]
        sub = (lbl[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea) + np.array([x, y])
        specks.append(Speck(float(cents[i][0]), float(cents[i][1]), float(longest[i] / 2),
                            float(contrast[i]), bool(signed[i] < 0), c))
    return specks


# ------------------------------------------------------------- annotation export
def _scaled_contour(c: np.ndarray, scale: float) -> np.ndarray:
    return np.round(c.reshape(-1, 2).astype(np.float32) / scale).astype(np.int32)


def save_annotation(src: Path, full_bgr: np.ndarray, mask_proc: np.ndarray,
                    det: BodyWings | None, specks: list[Speck], scale: float) -> Path:
    """Writes  <project>/src/<name>/ :
         <name>.<ext>          copy of the original image
         <name>_mask.png       full editable mask (brush + body/wing/dust you painted)
         wings_mask.png        detected wings
         dust_mask.png         the dust specks still kept in the list
         dust_instances.json   one entry per kept dust speck
    """
    out = output_dir_for(src)
    out.mkdir(parents=True, exist_ok=True)
    stem = clean_stem(src)

    copy = out / f"{stem}{src.suffix.lower()}"
    if not copy.exists():
        shutil.copy2(src, copy)

    H, W = full_bgr.shape[:2]

    def up(m: np.ndarray) -> np.ndarray:
        big = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        return ((big > 127) * 255).astype(np.uint8)

    imwrite(out / f"{stem}_mask.png", up(mask_proc))

    wings = np.zeros((H, W), np.uint8) if det is None else up(((det.wings > 0) * 255).astype(np.uint8))
    imwrite(out / "wings_mask.png", wings)

    dust = np.zeros((H, W), np.uint8)
    margin = max(1, round(5 / scale))
    instances = []
    for i, s in enumerate(specks, 1):
        c = _scaled_contour(s.contour, scale)
        cnt = c.reshape(-1, 1, 2)
        cv2.drawContours(dust, [cnt], -1, 255, cv2.FILLED)
        cv2.drawContours(dust, [cnt], -1, 255, margin)
        x, y, w, h = cv2.boundingRect(cnt)
        instances.append({
            "id": i,
            "center": [round(s.cx / scale, 1), round(s.cy / scale, 1)],
            "radius": round(s.r / scale, 1),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(cv2.contourArea(cnt)),
            "dark": bool(s.dark),
            "contrast": round(float(s.contrast), 2),
            "contour": c.tolist(),
        })
    imwrite(out / "dust_mask.png", dust)

    (out / "dust_instances.json").write_text(json.dumps({
        "image": src.name,
        "width": W,
        "height": H,
        "count": len(instances),
        "instances": instances,
    }, indent=2))
    return out


def load_dust(out_dir: Path, scale: float) -> list[Speck]:
    """Restore the dust list from a previous export (coordinates back to proxy space)."""
    p = out_dir / "dust_instances.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        specks = []
        for it in data["instances"]:
            c = (np.array(it["contour"], np.float32) * scale).round().astype(np.int32).reshape(-1, 1, 2)
            specks.append(Speck(it["center"][0] * scale, it["center"][1] * scale, it["radius"] * scale,
                                float(it["contrast"]), bool(it["dark"]), c))
        return specks
    except (KeyError, ValueError, TypeError):
        return []
