from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

try:                                    # optional: dust-suggestion helper from the other team
    from . import dust_assist
    ASSIST_ERROR = ""
except Exception as _ex:  # noqa: BLE001 - never let an optional helper stop the app from starting
    dust_assist, ASSIST_ERROR = None, f"{type(_ex).__name__}: {_ex}"
ASSIST_AVAILABLE = dust_assist is not None

MAX_SIDE = 2400  # detection / editing proxy size

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT if (_ROOT / "pyproject.toml").exists() else Path.cwd()
# Annotation output goes to  <project>/src/<image name>/
OUTPUT_ROOT = PROJECT_ROOT / "src"


def imread(path: str | Path) -> np.ndarray:
    """Decodes the image on the exact pixel grid stored in the file.

    IMREAD_IGNORE_ORIENTATION: OpenCV would otherwise silently rotate images that carry an EXIF
    orientation tag, so masks would no longer line up with the original file that is shipped in the
    benchmark (width/height would be swapped). The EXIF value is recorded in the metadata instead.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exif_orientation(path: str | Path) -> int:
    """EXIF orientation tag (1 = upright / none). Never raises."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.getexif().get(274, 1) or 1)
    except Exception:  # noqa: BLE001 - purely informational
        return 1


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


ANNOT_FILE = "annotation.json"          # project file: its presence marks an annotation folder
EDIT_MASK = "edit_mask.png"             # brush / filter mask (tool-internal, not part of the GT package)


def resolve_output_dir(src: Path) -> Path:
    """`src/<name>/`, made collision-proof: two *different* photos that share a file name
    (e.g. light/IMG_001.jpg and heavy/IMG_001.jpg) get `<name>` and `<name>_2` instead of
    overwriting each other. The folder name is the stable image ID of the benchmark."""
    base, sha = clean_stem(src), file_sha256(src)
    k = 1
    while True:
        d = OUTPUT_ROOT / (base if k == 1 else f"{base}_{k}")
        p = d / ANNOT_FILE
        if not p.exists():
            return d
        try:
            known = json.loads(p.read_text(encoding="utf-8")).get("source", {}).get("sha256")
        except (ValueError, OSError):
            known = None
        if known is None or known == sha:       # legacy project (no hash) or the very same photo
            return d
        k += 1


def load_mask(out_dir: Path, shape: tuple[int, int], stem: str) -> np.ndarray | None:
    for name in (EDIT_MASK, f"{stem}_mask.png"):            # second one = older versions
        p = out_dir / name
        if not p.exists():
            continue
        m = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return ((m > 127) * 255).astype(np.uint8)
    return None


# ====================================================================== classes
# Built-in classes = the four GT classes of the benchmark protocol (dust is handled as instances).
BODY_ID, WING_ID, INFO_ID = 1, 2, 3      # specimen / wings / paper-info (can be renamed, not deleted)
BUILTIN_IDS = (BODY_ID, WING_ID, INFO_ID)

# RGB colours handed out to newly created classes
PALETTE = [(251, 146, 60), (250, 204, 21), (74, 222, 128), (56, 189, 248), (244, 114, 182),
           (248, 113, 113), (163, 230, 53), (45, 212, 191), (148, 163, 184), (167, 139, 250)]


@dataclass
class ClassDef:
    id: int
    name: str
    color: tuple            # RGB
    visible: bool = True
    guard: bool = False     # dust detection keeps away from regions of this class


@dataclass(eq=False)        # identity comparison (numpy arrays inside)
class Region:
    """One editable polygon. `pts` are (N, 2) float32 in proxy-image coordinates."""
    cls: int
    pts: np.ndarray
    auto: bool = False      # True = produced by auto-detect and never touched by the user


def default_classes() -> list[ClassDef]:
    return [ClassDef(BODY_ID, "specimen", (45, 212, 191), guard=True),
            ClassDef(WING_ID, "wings", (139, 92, 246), guard=True),
            ClassDef(INFO_ID, "paper_info", (251, 146, 60), guard=True)]


_ZONE_OF_BUILTIN = {BODY_ID: "specimen", WING_ID: "wing", INFO_ID: "paper_info"}


def zone_name(c: ClassDef) -> str:
    """Canonical, rename-proof zone label used in dust_instances.json."""
    return _ZONE_OF_BUILTIN.get(c.id) or _safe(c.name).lower()


def ensure_defaults(classes: list[ClassDef]) -> bool:
    """Re-adds the built-in classes if missing. Returns True if the list changed."""
    have = {c.id for c in classes}
    changed = False
    for d in default_classes():
        if d.id not in have:
            classes.append(d)
            changed = True
    return changed


# ================================================================== detection
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


# ================================================== polygons (editable regions)
def _simplify_contour(c: np.ndarray, eps: float) -> np.ndarray:
    if eps > 0:
        c = cv2.approxPolyDP(c, eps, True)
    return c.reshape(-1, 2).astype(np.float32)


def regions_from_detection(det: BodyWings, eps: float = 3.0) -> list[Region]:
    """Turns the detector's masks into editable polygons (body class + wing class).
    `eps` = polygon simplification in px: bigger -> fewer nodes -> easier to edit by hand."""
    out: list[Region] = []
    bc, _ = cv2.findContours((det.body > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(bc, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) > 20:
            p = _simplify_contour(c, eps)
            if len(p) >= 3:
                out.append(Region(BODY_ID, p, auto=True))
    for lab in range(1, len(det.wing_contours) + 1):
        cs, _ = cv2.findContours((det.wings == lab).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        p = _simplify_contour(max(cs, key=cv2.contourArea), eps)
        if len(p) >= 3:
            out.append(Region(WING_ID, p, auto=True))
    return out


def fly_from_detection(det: BodyWings) -> np.ndarray:
    return (((det.body > 0) | (det.wings > 0)).astype(np.uint8)) * 255


def rasterize(regions: list[Region], shape: tuple[int, int], class_ids: set[int] | None = None) -> np.ndarray:
    """Filled union of the regions (optionally only some classes). uint8 0/255."""
    out = np.zeros(shape, np.uint8)
    for r in regions:
        if len(r.pts) >= 3 and (class_ids is None or r.cls in class_ids):
            cv2.fillPoly(out, [np.round(r.pts).astype(np.int32)], 255)
    return out


def point_in_poly(pts: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(pts.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), False) >= 0


def nearest_edge(pts: np.ndarray, x: float, y: float) -> tuple[float, int, np.ndarray]:
    """Closest point on the polygon outline -> (distance, index of edge start node, point)."""
    a = pts.astype(np.float32)
    ab = np.roll(a, -1, axis=0) - a
    ap = np.array([x, y], np.float32) - a
    t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    proj = a + ab * t[:, None]
    d = np.hypot(proj[:, 0] - x, proj[:, 1] - y)
    i = int(d.argmin())
    return float(d[i]), i, proj[i]


def subdivide(pts: np.ndarray) -> np.ndarray:
    """Adds a node in the middle of every edge."""
    mid = (pts + np.roll(pts, -1, axis=0)) / 2
    out = np.empty((len(pts) * 2, 2), np.float32)
    out[0::2], out[1::2] = pts, mid
    return out


def simplify_poly(pts: np.ndarray, eps: float) -> np.ndarray:
    """Removes nodes that deviate less than `eps` px. Keeps the polygon if it would collapse."""
    s = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.float32), eps, True).reshape(-1, 2)
    return s.astype(np.float32) if len(s) >= 3 else pts


def exclusion_zone(fly: np.ndarray | None, margin: float, shape: tuple[int, int]) -> np.ndarray:
    """`fly` (union of the guarded regions, gaps between them bridged) grown by `margin` px. uint8 0/255."""
    h, w = shape
    zone = np.zeros((h, w), np.uint8)
    if fly is None or not fly.any():
        return zone
    fly = (fly > 0).astype(np.uint8) * 255
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
    uid: int = 0                    # stable per-image number -> "MOTA_<uid>" (0 = not assigned yet)
    cls: int | None = None          # class chosen by hand; None = derived from where the speck sits
    manual: bool = False            # added by hand (kept when dust detection is re-run)
    source: str = ""                # "dust_assist" = accepted from the optional helper's suggestions


def ensure_uids(specks: list[Speck]) -> list[Speck]:
    """Gives every speck a unique, stable id. Existing ids are kept (deleting a speck leaves a gap
    instead of renumbering the rest); missing / duplicated ones get the next free number."""
    seen: set[int] = set()
    out: list[Speck] = []
    nxt = max((s.uid for s in specks), default=0) + 1
    for s in specks:
        if s.uid <= 0 or s.uid in seen:
            s = replace(s, uid=nxt)
            nxt += 1
        seen.add(s.uid)
        out.append(s)
    return out


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
    specks.sort(key=lambda sp: (round(sp.cy), sp.cx))        # reading order -> MOTA_001 is top-left
    return [replace(sp, uid=k) for k, sp in enumerate(specks, 1)]


def speck_at(bgr: np.ndarray, x: float, y: float, max_diam: int = 20, fallback_r: float = 4.0) -> Speck:
    """Builds ONE speck at a clicked position (manual annotation of dust the detector missed).
    Tries to outline the real blob under the cursor; falls back to a small circle."""
    H, W = bgr.shape[:2]
    half = int(max(12, max_diam * 2.5))
    x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
    x1, y1 = min(W, int(x) + half + 1), min(H, int(y) + half + 1)
    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    local = cv2.medianBlur(gray, _odd(max(7, max_diam * 2.5)))
    diff = gray.astype(np.int16) - local.astype(np.int16)
    ad = np.abs(diff)
    cx, cy = int(round(x)) - x0, int(round(y)) - y0
    ya, xa = max(0, cy - 4), max(0, cx - 4)
    sub = ad[ya:cy + 5, xa:cx + 5]
    if sub.size and sub.max() >= 4:
        py, px = np.unravel_index(int(sub.argmax()), sub.shape)
        py, px = py + ya, px + xa
        mask = (ad > max(3.0, 0.5 * float(ad[py, px]))).astype(np.uint8)
        n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        i = int(lbl[py, px])
        if i > 0:
            bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            if stats[i, cv2.CC_STAT_AREA] >= 3 and max(bw, bh) <= max_diam:
                xx, yy = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
                sub_m = (lbl[yy:yy + bh, xx:xx + bw] == i).astype(np.uint8)
                cs, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    c = max(cs, key=cv2.contourArea) + np.array([xx + x0, yy + y0])
                    sel = lbl == i
                    return Speck(float(cents[i][0] + x0), float(cents[i][1] + y0), max(bw, bh) / 2,
                                 float(ad[sel].mean()), bool(diff[sel].sum() < 0), c, manual=True)
    r = max(2.0, float(fallback_r))
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    c = np.stack([x + r * np.cos(t), y + r * np.sin(t)], 1).round().astype(np.int32).reshape(-1, 1, 2)
    return Speck(float(x), float(y), r, 0.0, True, c, manual=True)


def assign_zones(specks: list[Speck], regions: list[Region], classes: list[ClassDef],
                 shape: tuple[int, int]) -> list[int | None]:
    """Class each speck sits on: the manual choice if there is one, otherwise the class whose
    regions overlap the speck the most. None = background (no class under it)."""
    valid = {c.id for c in classes}
    masks = {}
    for c in classes:
        m = rasterize(regions, shape, {c.id})
        if m.any():
            masks[c.id] = m > 0
    H, W = shape
    out: list[int | None] = []
    for s in specks:
        if s.cls is not None and s.cls in valid:
            out.append(s.cls)
            continue
        if not masks:
            out.append(None)
            continue
        poly = np.round(_speck_poly(s)).astype(np.int32)
        x0, y0 = np.clip(poly.min(0), 0, (W - 1, H - 1))
        x1, y1 = np.clip(poly.max(0) + 1, 1, (W, H))
        if x1 <= x0 or y1 <= y0:
            out.append(None)
            continue
        loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(loc, [poly - (x0, y0)], 1)
        loc[min(max(int(round(s.cy)) - y0, 0), loc.shape[0] - 1),
            min(max(int(round(s.cx)) - x0, 0), loc.shape[1] - 1)] = 1      # tiny contours: centre counts
        best, best_n = None, 0
        for cid, m in masks.items():
            n = int(np.count_nonzero(m[y0:y1, x0:x1] & (loc > 0)))
            if n > best_n:
                best, best_n = cid, n
        out.append(best)
    return out


# ------------------------------------------- optional: dust suggestions (dust_assist helper)
# The helper is a detection-only SECOND OPINION. Its output lives in its own layer and is never
# ground truth: a suggestion only becomes dust when a person accepts it, and then it is an ordinary
# hand-curated speck (origin "manual", plus "suggested_by" in dust_instances.json for traceability).
SUGGESTIONS_JSON = "dust_suggestions.json"      # provenance file - deliberately NOT in GT_FILES
ASSIST_SOURCE = "dust_assist"


@dataclass(frozen=True, eq=False)               # numpy inside -> identity comparison
class Suggestion:
    uid: int
    cx: float                       # proxy-image coordinates (like Speck)
    cy: float
    r: float
    contrast: float                 # the helper's peak contrast (not the same scale as Speck.contrast)
    dark: bool
    contour: np.ndarray             # (N, 2) float32, proxy coordinates (kept unrounded)
    zone: str                       # the helper's own zone: "background" | "paper_info"
    status: str                     # the helper's own verdict: "candidate" | "review"
    reason: str
    recurrent: bool = False
    decision: str = "pending"       # pending | accepted | rejected  - only a person changes this


def assist_suggestions(bgr: np.ndarray, specimen: np.ndarray | None = None,
                       document: np.ndarray | None = None, recurrent: np.ndarray | None = None,
                       include_paper: bool = True, sensitivity: float = 5.0,
                       to_proc: float = 1.0):
    """Runs the helper on `bgr` -> (suggestions in reading order, the helper's DetectionResult).

    specimen / document: uint8 masks of the SAME size as `bgr`; None = the helper guesses its own.
    recurrent: (N, 2) sensor-dust positions in `bgr` pixels.
    sensitivity: 1..10 like the other sliders (5 = the helper's own defaults, higher = more suggestions).
    to_proc: factor from `bgr` pixels to proxy pixels (1.0 when `bgr` already is the proxy)."""
    if dust_assist is None:
        raise RuntimeError("the dust_assist helper is not available")
    k = 7.5 - 0.5 * float(sensitivity)                 # 5 -> 5.0 (default), 10 -> 2.5, 1 -> 7.0
    params = dust_assist.Params(dust_sigma_k=k, min_dust_contrast=2.0 * k)
    if not include_paper:
        document = np.zeros(bgr.shape[:2], np.uint8)   # an empty paper mask = no paper / label proposals
    res = dust_assist.detect_suggestions(bgr, specimen_mask=specimen, document_mask=document,
                                         recurrent_positions=recurrent, params=params)
    out: list[Suggestion] = []
    for d in res.dust:
        x, y, w, h = d.bbox
        c = np.asarray(d.contour or [], np.float32).reshape(-1, 2)
        if len(c) == 0:                                # no contour came back: keep a tiny box
            c = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        out.append(Suggestion(0, d.center[0] * to_proc, d.center[1] * to_proc,
                              max(1.0, 0.5 * max(w, h) * to_proc), float(d.contrast), bool(d.dark),
                              c * np.float32(to_proc), d.zone, d.status, d.reason, bool(d.recurrent)))
    out.sort(key=lambda s: (round(s.cy), s.cx))        # reading order, like detect_dust
    return [replace(s, uid=n) for n, s in enumerate(out, 1)], res


def speck_from_suggestion(s: Suggestion) -> Speck:
    """The speck that exists once a person accepts a suggestion (keeps the provenance)."""
    cnt = np.round(s.contour).astype(np.int32).reshape(-1, 1, 2)
    return Speck(s.cx, s.cy, s.r, s.contrast, s.dark, cnt, manual=True, source=ASSIST_SOURCE)


def regions_from_mask(mask: np.ndarray, class_id: int, eps: float = 3.0,
                      min_area: float = 20.0) -> list[Region]:
    """Editable polygons (auto=True) from a binary mask, largest first."""
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Region] = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) > min_area:
            p = _simplify_contour(c, eps)
            if len(p) >= 3:
                out.append(Region(class_id, p, auto=True))
    return out


def assist_document_regions(bgr: np.ndarray, eps: float = 3.0) -> list[Region]:
    """Starting point for the paper / info class: light, rectangular regions found by the helper.
    The polygons are `auto` (replaced by the next proposal until a person edits them)."""
    if dust_assist is None:
        raise RuntimeError("the dust_assist helper is not available")
    p = dust_assist.Params()
    background = dust_assist.estimate_background(bgr, p)[0]
    doc = dust_assist.detect_document(bgr, background, p)
    return regions_from_mask(doc.astype(np.uint8) * 255, INFO_ID, eps)


def assist_recurrent_positions(proxies: list[np.ndarray], min_occurrences: int = 3) -> np.ndarray:
    """Sensor-dust positions that repeat across photos of one session (all of the same size)."""
    if dust_assist is None:
        raise RuntimeError("the dust_assist helper is not available")
    return dust_assist.recurring_positions(proxies, min_occurrences=min_occurrences)


def write_suggestions(out: Path, src: Path, shape_hw: tuple[int, int], sugs: list[Suggestion],
                      scale: float) -> Path:
    """`dust_suggestions.json`: what the helper proposed and what the person decided.
    Provenance only: it is NOT ground truth and is not part of GT_FILES (so never packaged)."""
    H, W = shape_hw
    inst = []
    for s in sugs:
        cont = _scaled_contour(s.contour, scale)
        cnt = cont.reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        inst.append({
            "id": f"SUG_{s.uid:03d}",
            "decision": s.decision,
            "center": [round(s.cx / scale, 1), round(s.cy / scale, 1)],
            "radius": round(s.r / scale, 1),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(cv2.contourArea(cnt)),
            "dark": bool(s.dark),
            "contrast": round(float(s.contrast), 2),
            "contour": cont.tolist(),
            "zone": s.zone, "status": s.status, "reason": s.reason, "recurrent": bool(s.recurrent),
        })
    path = out / SUGGESTIONS_JSON
    path.write_text(json.dumps({
        "image_id": out.name,
        "image": src.name,
        "width": W,
        "height": H,
        "coordinates": "original-image pixels, origin top-left",
        "source": "dust_assist (detection-only helper)",
        "ground_truth": False,
        "count": len(inst),
        "instances": inst,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------------- annotation export
def _scaled_contour(c: np.ndarray, scale: float) -> np.ndarray:
    return np.round(c.reshape(-1, 2).astype(np.float32) / scale).astype(np.int32)


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "class"


def _speck_poly(s: Speck) -> np.ndarray:
    """Contour of a speck, or a small box when the contour is too small to be a polygon."""
    p = s.contour.reshape(-1, 2).astype(np.float32)
    if len(p) >= 3:
        return p
    r = max(1.0, s.r)
    return np.array([[s.cx - r, s.cy - r], [s.cx + r, s.cy - r],
                     [s.cx + r, s.cy + r], [s.cx - r, s.cy + r]], np.float32)


def dust_id(uid: int) -> str:
    return f"MOTA_{uid:03d}"


# Files of the GT package (names follow the sprint-1 protocol).
SPECIMEN_MASK, WINGS_MASK, DUST_MASK, INFO_MASK = ("specimen_mask.png", "wings_mask.png",
                                                   "dust_mask.png", "info_mask.png")
DUST_JSON = "dust_instances.json"
GT_FILES = (SPECIMEN_MASK, WINGS_MASK, DUST_MASK, INFO_MASK, DUST_JSON)


def save_annotation(src: Path, out: Path, full_bgr: np.ndarray, mask_proc: np.ndarray,
                    classes: list[ClassDef], regions: list[Region], specks: list[Speck],
                    scale: float, proc_shape: tuple[int, int]) -> Path:
    """Writes the annotation folder  <project>/src/<image id>/  :

         original.<ext>        byte-exact copy of the source photo
         specimen_mask.png     \\
         wings_mask.png         |  full-resolution, single channel, same width x height as the original,
         dust_mask.png          |  255 = class, 0 = not
         info_mask.png         /
         dust_instances.json   one entry per dust speck: id, class / zone, actionable, geometry
         extra_classes/        one mask per user-defined class (only if any exist)
         annotation.json       editable project (classes + polygons, normalized) - reloaded on open
         edit_mask.png         brush / filter mask of the editor (tool-internal)

       (dust_suggestions.json is written separately by `write_suggestions`, only when asked: it is
        provenance of the optional dust-assist helper - NOT ground truth and never packaged)
    """
    out.mkdir(parents=True, exist_ok=True)
    stem = clean_stem(src)
    image_id = out.name
    H, W = full_bgr.shape[:2]
    ph, pw = proc_shape
    proc_wh = np.array([pw, ph], np.float32)
    full_wh = np.array([W, H], np.float32)

    # ---- original (copy)
    dest = out / f"original{src.suffix.lower()}"
    for old in out.glob("original.*"):
        if old != dest:
            old.unlink()
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)

    def norm(p: np.ndarray) -> np.ndarray:             # proxy px -> 0..1
        return np.clip(np.asarray(p, np.float32).reshape(-1, 2) / proc_wh, 0.0, 1.0)

    def to_px(n: np.ndarray) -> np.ndarray:            # 0..1 -> full-res px polygon for cv2
        return np.round(n * full_wh).astype(np.int32)

    def up(m: np.ndarray) -> np.ndarray:
        big = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        return ((big > 127) * 255).astype(np.uint8)

    # ---- classes -> full-res masks (polygons are rasterized straight at full resolution)
    by_id = {c.id: c for c in classes}
    class_masks = {c.id: np.zeros((H, W), np.uint8) for c in classes}
    reg_json, inst_no = [], {}
    for r in regions:
        c = by_id.get(r.cls)
        if c is None or len(r.pts) < 3:
            continue
        n = norm(r.pts)
        cv2.fillPoly(class_masks[c.id], [to_px(n)], 255)
        inst_no[c.id] = inst_no.get(c.id, 0) + 1
        reg_json.append({"class_id": c.id, "class": c.name, "instance": inst_no[c.id],
                         "auto": bool(r.auto), "points": np.round(n, 6).tolist()})

    empty = np.zeros((H, W), np.uint8)
    imwrite(out / SPECIMEN_MASK, class_masks.get(BODY_ID, empty))
    imwrite(out / WINGS_MASK, class_masks.get(WING_ID, empty))
    imwrite(out / INFO_MASK, class_masks.get(INFO_ID, empty))
    extra = out / "extra_classes"
    if extra.exists():
        shutil.rmtree(extra)
    for c in classes:
        if c.id not in BUILTIN_IDS and class_masks[c.id].any():
            extra.mkdir(exist_ok=True)
            imwrite(extra / f"{_safe(c.name)}.png", class_masks[c.id])

    imwrite(out / EDIT_MASK, up(mask_proc))

    # ---- dust: global mask + one instance per speck, each tied to the class it sits on
    specks = sorted(ensure_uids(specks), key=lambda sp: sp.uid)
    zones = assign_zones(specks, regions, classes, proc_shape)
    dust = np.zeros((H, W), np.uint8)
    margin = max(1, round(5 / scale))
    instances = []
    for s, zc in zip(specks, zones):
        c = by_id.get(zc) if zc is not None else None
        cont = _scaled_contour(s.contour, scale)
        cnt = cont.reshape(-1, 1, 2)
        cv2.drawContours(dust, [cnt], -1, 255, cv2.FILLED)
        cv2.drawContours(dust, [cnt], -1, 255, margin)
        x, y, w, h = cv2.boundingRect(cnt)
        instances.append({
            "id": dust_id(s.uid),
            "class_id": c.id if c else None,
            "class": c.name if c else "background",
            "zone": zone_name(c) if c else "background",
            "origin": "manual" if s.manual else "auto",
            "class_source": "manual" if (s.cls is not None and s.cls in by_id) else "auto",
            "actionable": bool(c is None or not c.guard),
            "center": [round(s.cx / scale, 1), round(s.cy / scale, 1)],
            "radius": round(s.r / scale, 1),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(cv2.contourArea(cnt)),
            "dark": bool(s.dark),
            "contrast": round(float(s.contrast), 2),
            "contour": cont.tolist(),
        })
        if s.source:                                   # accepted from the optional dust-assist helper
            instances[-1]["suggested_by"] = s.source
    imwrite(out / DUST_MASK, dust)
    (out / DUST_JSON).write_text(json.dumps({
        "image_id": image_id,
        "image": dest.name,
        "width": W,
        "height": H,
        "coordinates": "original-image pixels, origin top-left",
        "count": len(instances),
        "count_actionable": sum(d["actionable"] for d in instances),
        "instances": instances,
    }, indent=2), encoding="utf-8")

    # ---- editable project
    (out / ANNOT_FILE).write_text(json.dumps({
        "version": 3,
        "image_id": image_id,
        "image": src.name, "width": W, "height": H,
        "source": {"file": src.name, "folder": src.parent.name, "sha256": file_sha256(src),
                   "exif_orientation": exif_orientation(src)},
        "classes": [{"id": c.id, "name": c.name, "color": list(c.color),
                     "visible": c.visible, "guard": c.guard} for c in classes],
        "regions": reg_json,
    }, indent=2), encoding="utf-8")

    # ---- remove files written by older versions of the tool (labels / per-class dump / copies)
    for legacy in (out / "class_masks", ):
        if legacy.exists():
            shutil.rmtree(legacy)
    for name in ("classes.txt", f"{stem}.txt", f"{stem}_mask.png"):
        (out / name).unlink(missing_ok=True)
    for ext in _IMG_EXTS:
        old = out / f"{stem}{ext}"
        if old != dest:
            old.unlink(missing_ok=True)
    return out


def load_project(out_dir: Path, proc_shape: tuple[int, int]) -> tuple[list[ClassDef], list[Region]] | None:
    """Restores classes + editable regions saved by `save_annotation` (None if there are none)."""
    p = out_dir / ANNOT_FILE
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ph, pw = proc_shape
        wh = np.array([pw, ph], np.float32)
        classes = [ClassDef(int(c["id"]), str(c["name"]), tuple(int(v) for v in c["color"]),
                            bool(c.get("visible", True)), bool(c.get("guard", False)))
                   for c in data["classes"]]
        ids = {c.id for c in classes}
        regions = []
        for it in data["regions"]:
            if int(it["class_id"]) not in ids:
                continue
            pts = (np.array(it["points"], np.float32).reshape(-1, 2) * wh).astype(np.float32)
            if len(pts) >= 3:
                regions.append(Region(int(it["class_id"]), pts, bool(it.get("auto", False))))
        ensure_defaults(classes)
        return classes, regions
    except (KeyError, ValueError, TypeError):
        return None


def load_dust(out_dir: Path, scale: float, class_ids: set[int] | None = None) -> list[Speck]:
    """Restore the dust list from a previous export (coordinates back to proxy space)."""
    p = out_dir / DUST_JSON
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        specks = []
        for k, it in enumerate(data["instances"], 1):
            c = (np.array(it["contour"], np.float32) * scale).round().astype(np.int32).reshape(-1, 1, 2)
            raw = it.get("id", k)
            uid = int(str(raw).split("_")[-1]) if not isinstance(raw, int) else raw
            cls = it.get("class_id") if it.get("class_source") == "manual" else None
            if cls is not None and class_ids is not None and int(cls) not in class_ids:
                cls = None
            specks.append(Speck(it["center"][0] * scale, it["center"][1] * scale, it["radius"] * scale,
                                float(it["contrast"]), bool(it["dark"]), c, uid,
                                int(cls) if cls is not None else None,
                                it.get("origin") == "manual",
                                source=str(it.get("suggested_by") or "")))
        return ensure_uids(specks)
    except (KeyError, ValueError, TypeError):
        return []
