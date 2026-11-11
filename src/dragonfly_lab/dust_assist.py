"""Dust/background annotation assistant for Dragonfly Lab.

Refactored from the Team 1 ``LiberulasDummy`` prototype.  This module is
INTENTIONALLY detection-only: it never inpaints, deletes, or modifies source
pixels.  Its output is meant to be reviewed/corrected by a human annotator.

Dependencies: numpy, opencv-python (or opencv-python-headless)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Params:
    # General background / dust
    background_kernel_frac: float = 0.12
    background_work_max_side: int = 600
    object_sigma_k: float = 3.0
    dust_sigma_k: float = 5.0
    recurrent_dust_sigma_k: float = 3.0
    object_close_frac: float = 0.035
    min_object_area_frac: float = 0.004
    max_dust_area_frac: float = 0.0015
    min_dust_area_px: int = 2
    protection_margin_frac: float = 0.015
    sigma_floor: float = 1.5
    min_object_contrast: float = 8.0
    global_object_delta_lab: float = 18.0
    min_dust_contrast: float = 10.0
    min_recurrent_dust_contrast: float = 6.0
    border_px: int = 2
    recurrence_tolerance_frac: float = 0.012

    # Paper / label / millimetric sheet helper
    detect_document: bool = True
    document_min_area_frac: float = 0.02
    document_min_extent: float = 0.6
    document_brightness_delta: float = 15.0
    document_L_min: float = 170.0
    document_local_background_frac: float = 0.08
    document_ink_delta: float = 12.0
    document_min_dust_contrast: float = 20.0
    document_max_dust_chroma: float = 12.0
    document_max_dust_area_frac: float = 0.0008
    document_text_isolation_frac: float = 0.03
    document_border_margin_frac: float = 0.02


@dataclass
class DustSuggestion:
    id: int
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    contrast: float
    dark: bool
    zone: str
    status: str
    reason: str
    recurrent: bool = False
    contour: list[list[int]] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    dust: list[DustSuggestion]
    document_mask: np.ndarray
    auto_object_mask: np.ndarray
    protection_mask: np.ndarray
    information_mask: np.ndarray
    sigma: float


def _odd(n: float, minimum: int = 3) -> int:
    n = max(int(round(n)), minimum)
    return n if n % 2 else n + 1


def _components(binary: np.ndarray):
    return cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)


def _robust_sigma(values: np.ndarray, floor: float) -> float:
    if values.size == 0:
        return floor
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return max(float(1.4826 * mad), floor)


def _fill_small_holes(mask: np.ndarray, max_hole_frac: float = 0.02) -> np.ndarray:
    h, w = mask.shape
    n, labels, stats, _ = _components(~mask)
    out = mask.copy()
    limit = max_hole_frac * h * w
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        touches = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not touches and area <= limit:
            out[labels == i] = True
    return out


def _scaled_median(gray: np.ndarray, p: Params) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, p.background_work_max_side / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    k = _odd(p.background_kernel_frac * min(small.shape))
    bg = cv2.medianBlur(small.astype(np.uint8), k)
    if scale < 1:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg.astype(np.float32)


def _polynomial_surface(channels: np.ndarray, p: Params) -> np.ndarray:
    """Degree-2 Lab surface: captures smooth illumination/background, not objects."""
    h, w, c = channels.shape
    step = max(1, int(np.sqrt(h * w / 40000)))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    xn, yn = xs.ravel() / w - 0.5, ys.ravel() / h - 0.5
    A = np.stack([np.ones_like(xn), xn, yn, xn*xn, yn*yn, xn*yn], 1)
    samples = channels[::step, ::step].reshape(-1, c).astype(np.float64)

    q = np.floor(samples / 6).astype(np.int64)
    keys = q[:, 0] * 10**6 + q[:, 1] * 10**3 + q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    mode = samples[keys == values[np.argmax(counts)]].mean(0)
    use = np.linalg.norm(samples - mode, axis=1) < p.global_object_delta_lab
    if use.sum() < 50:
        use[:] = True

    coef = None
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[use], samples[use], rcond=None)
        dist = np.linalg.norm(samples - A @ coef, axis=1)
        s = _robust_sigma(dist[use], p.sigma_floor)
        new_use = dist < max(3 * s, p.min_object_contrast)
        if new_use.sum() < 50:
            break
        use = new_use

    hs, ws = min(h, 400), min(w, 400)
    Y, X = np.mgrid[0:hs, 0:ws].astype(np.float32)
    X, Y = (X + 0.5) / ws - 0.5, (Y + 0.5) / hs - 0.5
    terms = [np.ones_like(X), X, Y, X*X, Y*Y, X*Y]
    surface = np.stack([sum(t * coef[i, k] for i, t in enumerate(terms)) for k in range(c)], -1)
    if (hs, ws) != (h, w):
        surface = cv2.resize(surface, (w, h), interpolation=cv2.INTER_LINEAR)
    return surface.astype(np.float32)


def estimate_background(image_bgr: np.ndarray, p: Params):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    surface = _polynomial_surface(lab, p)
    color_distance = np.linalg.norm(lab - surface, axis=2)
    approx_object = color_distance > p.global_object_delta_lab
    approx_object = cv2.dilate(approx_object.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    L = lab[..., 0]
    filled = L.copy()
    filled[approx_object] = surface[..., 0][approx_object]
    background = _scaled_median(np.clip(filled, 0, 255).astype(np.uint8), p)
    residual = L - background
    free = ~approx_object
    sigma = _robust_sigma(residual[free] if free.sum() > 100 else residual, p.sigma_floor)
    return background, residual, color_distance, sigma


def _object_threshold(sigma: float, p: Params) -> float:
    return max(p.object_sigma_k * sigma, p.min_object_contrast)


def _dust_threshold(sigma: float, p: Params, recurrent: bool = False) -> float:
    if recurrent:
        return max(p.recurrent_dust_sigma_k * sigma, p.min_recurrent_dust_contrast)
    return max(p.dust_sigma_k * sigma, p.min_dust_contrast)


def auto_object_and_protection(residual: np.ndarray, color_distance: np.ndarray, sigma: float, p: Params):
    h, w = residual.shape
    raw = (np.abs(residual) > _object_threshold(sigma, p)) | (color_distance > p.global_object_delta_lab)

    scale = min(1.0, 1500 / max(h, w))
    small = cv2.resize(raw.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else raw.astype(np.uint8)
    small = (small > 0).astype(np.uint8)
    k = _odd(p.object_close_frac * min(small.shape))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if scale < 1:
        closed = cv2.resize(closed, (w, h), interpolation=cv2.INTER_NEAREST)
    closed = (closed > 0) | raw

    n, labels, stats, _ = _components(closed)
    min_area = p.min_object_area_frac * h * w
    obj = np.zeros((h, w), bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            obj |= labels == i
    obj = _fill_small_holes(obj)

    margin = _odd(p.protection_margin_frac * min(h, w))
    protection = cv2.dilate(obj.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))) > 0
    return obj, protection


def _expand_protection(mask: np.ndarray, p: Params) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask > 0
    margin = _odd(p.protection_margin_frac * min(mask.shape))
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))) > 0


def detect_document(image_bgr: np.ndarray, background: np.ndarray, p: Params) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    bright = (L > background + p.document_brightness_delta) & (L > p.document_L_min)
    k = _odd(0.03 * min(h, w))
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) > 0
    n, labels, stats, _ = _components(bright)
    doc = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area >= p.document_min_area_frac * h * w and area / max(1, ww * hh) >= p.document_min_extent:
            doc |= labels == i
    return _fill_small_holes(doc, 0.05) if doc.any() else doc


def _component_contour(labels: np.ndarray, component: int, bbox: tuple[int, int, int, int]) -> list[list[int]]:
    x, y, w, h = bbox
    crop = (labels[y:y+h, x:x+w] == component).astype(np.uint8) * 255
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return [[int(px + x), int(py + y)] for px, py in cnt]


def detect_background_dust(residual: np.ndarray, sigma: float, object_mask: np.ndarray,
                           protection_mask: np.ndarray, p: Params,
                           recurrent_positions: np.ndarray | None = None) -> list[DustSuggestion]:
    h, w = residual.shape
    candidate_threshold = min(_dust_threshold(sigma, p, True), _object_threshold(sigma, p))
    candidate = (np.abs(residual) > candidate_threshold) & ~object_mask
    n, labels, stats, centroids = _components(candidate)
    max_area = max(p.max_dust_area_frac * h * w, 4)
    tolerance = p.recurrence_tolerance_frac * min(h, w)
    out: list[DustSuggestion] = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 1:
            continue
        x, y, ww, hh = (int(v) for v in stats[i, :4])
        sl = (slice(y, y + hh), slice(x, x + ww))
        region = labels[sl] == i
        values = residual[sl][region]
        peak = float(values[np.argmax(np.abs(values))])
        contrast = abs(peak)
        core_area = int((np.abs(values) >= 0.5 * contrast).sum())
        cx, cy = map(float, centroids[i])

        recurrent = False
        if recurrent_positions is not None and len(recurrent_positions):
            d = np.hypot(recurrent_positions[:, 0] - cx, recurrent_positions[:, 1] - cy)
            recurrent = bool(d.min() <= tolerance)

        if contrast < _dust_threshold(sigma, p, recurrent):
            continue
        if area < p.min_dust_area_px and not recurrent:
            continue

        touches_border = (x <= p.border_px or y <= p.border_px or
                          x + ww >= w - p.border_px or y + hh >= h - p.border_px)
        if (region & protection_mask[sl]).any():
            status, reason = "review", "touches protected specimen/support region"
        elif touches_border:
            status, reason = "review", "touches image border"
        elif core_area > max_area:
            status, reason = "review", "too large for an automatic dust suggestion"
        else:
            status = "candidate"
            reason = "recurrent sensor-dust candidate" if recurrent else "isolated background speck"

        contour = _component_contour(labels, i, (x, y, ww, hh))
        out.append(DustSuggestion(
            id=len(out) + 1,
            center=(round(cx, 1), round(cy, 1)),
            bbox=(x, y, ww, hh),
            area=area,
            contrast=round(contrast, 2),
            dark=peak < 0,
            zone="background",
            status=status,
            reason=reason,
            recurrent=recurrent,
            contour=contour,
        ))
    return out


def detect_document_dust(image_bgr: np.ndarray, document_mask: np.ndarray,
                         specimen_mask: np.ndarray, p: Params,
                         first_id: int = 1):
    """Conservative candidates on paper/label; isolated from text/grid strokes."""
    h, w = document_mask.shape
    if not document_mask.any():
        return [], np.zeros((h, w), bool)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.hypot(a, b)
    Lu = L.astype(np.uint8).copy()
    Lu[~document_mask] = int(np.median(L[document_mask]))

    scale = min(1.0, 1200 / max(h, w))
    small = cv2.resize(Lu, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else Lu
    paper_bg = cv2.medianBlur(small, _odd(p.document_local_background_frac * min(small.shape), 5))
    if scale < 1:
        paper_bg = cv2.resize(paper_bg, (w, h), interpolation=cv2.INTER_LINEAR)
    paper_bg = paper_bg.astype(np.float32)

    darkness = paper_bg - L
    ink_all = (darkness > p.document_ink_delta) & document_mask
    grid = ink_all & (a > 6) & (b > 6) & (chroma > 15)
    ink = ink_all & ~grid

    near_specimen = _expand_protection(specimen_mask, p)
    edge_k = _odd(p.document_border_margin_frac * min(h, w))
    interior = cv2.erode(document_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (edge_k, edge_k))) > 0
    isolation = int(np.ceil(p.document_text_isolation_frac * min(h, w)))
    max_area = max(p.document_max_dust_area_frac * h * w, 4)

    n, labels, stats, centroids = _components(ink)
    suggestions: list[DustSuggestion] = []
    information = ink_all.copy()
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if area < p.min_dust_area_px or area > 4 * max_area:
            continue
        if max(ww, hh) / max(1, min(ww, hh)) > 3:
            continue
        sl = (slice(y, y + hh), slice(x, x + ww))
        region = labels[sl] == i
        vals = darkness[sl][region]
        k = int(np.argmax(vals))
        contrast = float(vals[k])
        if contrast < p.document_min_dust_contrast or float(chroma[sl][region][k]) > p.document_max_dust_chroma:
            continue
        if int((vals >= 0.5 * contrast).sum()) > max_area:
            continue

        ya, yb = max(0, y - isolation), min(h, y + hh + isolation)
        xa, xb = max(0, x - isolation), min(w, x + ww + isolation)
        neighbors = ink[ya:yb, xa:xb] & (labels[ya:yb, xa:xb] != i)
        if neighbors.any():
            continue

        if near_specimen[sl][region].any():
            status, reason = "review", "paper speck near specimen"
        elif not interior[sl][region].all():
            status, reason = "review", "paper speck near document border"
        else:
            status, reason = "candidate", "isolated low-chroma speck on paper/label"
            information[sl][region] = False

        cx, cy = map(float, centroids[i])
        contour = _component_contour(labels, i, (x, y, ww, hh))
        suggestions.append(DustSuggestion(
            id=first_id + len(suggestions),
            center=(round(cx, 1), round(cy, 1)),
            bbox=(x, y, ww, hh),
            area=int(area),
            contrast=round(contrast, 2),
            dark=True,
            zone="paper_info",
            status=status,
            reason=reason,
            recurrent=False,
            contour=contour,
        ))
    return suggestions, information


def detect_suggestions(image_bgr: np.ndarray,
                       specimen_mask: np.ndarray | None = None,
                       document_mask: np.ndarray | None = None,
                       recurrent_positions: np.ndarray | None = None,
                       params: Params | None = None) -> DetectionResult:
    """Main entry point for an annotation UI.

    ``specimen_mask`` should be the current human-corrected body+wings mask from
    Dragonfly Lab when available.  If absent, a conservative automatic object mask
    is used only to protect the specimen from dust proposals.

    No output here is Ground Truth.  Every suggestion must remain editable/rejectable.
    """
    p = params or Params()
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a BGR image with shape HxWx3")

    background, residual, color_distance, sigma = estimate_background(image_bgr, p)
    auto_object, auto_protection = auto_object_and_protection(residual, color_distance, sigma, p)

    if specimen_mask is not None:
        if specimen_mask.shape[:2] != image_bgr.shape[:2]:
            raise ValueError("specimen_mask must have the same HxW as image_bgr")
        specimen = specimen_mask > 0
        protection = _expand_protection(specimen, p)
        # Keep auto-object only as an additional conservative guard against missed specimen/support.
        protection |= auto_protection
        object_for_background = specimen | auto_object
    else:
        specimen = auto_object
        protection = auto_protection
        object_for_background = auto_object

    if document_mask is None:
        document = detect_document(image_bgr, background, p) if p.detect_document else np.zeros(image_bgr.shape[:2], bool)
    else:
        if document_mask.shape[:2] != image_bgr.shape[:2]:
            raise ValueError("document_mask must have the same HxW as image_bgr")
        document = document_mask > 0

    bg_dust = detect_background_dust(residual, sigma, object_for_background, protection, p, recurrent_positions)
    doc_dust, info_mask = detect_document_dust(image_bgr, document, specimen, p, first_id=len(bg_dust) + 1)
    return DetectionResult(
        dust=bg_dust + doc_dust,
        document_mask=document,
        auto_object_mask=auto_object,
        protection_mask=protection,
        information_mask=info_mask,
        sigma=sigma,
    )


def recurring_positions(images_bgr: Iterable[np.ndarray], p: Params | None = None,
                        min_occurrences: int = 3) -> np.ndarray:
    """Optional session helper: candidates repeatedly appearing at the same sensor coordinates.

    All images must share dimensions.  Use only as a confidence hint; human review remains required.
    """
    p = p or Params()
    images = list(images_bgr)
    if not images:
        return np.zeros((0, 2), dtype=np.float32)
    h, w = images[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in images):
        raise ValueError("all session images must have identical dimensions")

    summed = np.zeros((h, w), np.float64)
    count = np.zeros((h, w), np.int32)
    for image in images:
        _, residual, color_distance, sigma = estimate_background(image, p)
        _, protection = auto_object_and_protection(residual, color_distance, sigma, p)
        free = ~protection
        summed[free] += residual[free]
        count[free] += 1

    valid = count >= min_occurrences
    mean = np.zeros((h, w), np.float32)
    mean[valid] = (summed[valid] / count[valid]).astype(np.float32)
    threshold = p.min_recurrent_dust_contrast
    candidates = (np.abs(mean) > threshold * 0.75) & valid
    n, labels, stats, centroids = _components(candidates)
    max_area = p.max_dust_area_frac * h * w
    points = []
    for i in range(1, n):
        x, y, ww, hh = (int(v) for v in stats[i, :4])
        vals = mean[y:y+hh, x:x+ww][labels[y:y+hh, x:x+ww] == i]
        if stats[i, cv2.CC_STAT_AREA] <= max_area and vals.size and np.abs(vals).max() >= threshold:
            points.append(centroids[i])
    return np.asarray(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)
