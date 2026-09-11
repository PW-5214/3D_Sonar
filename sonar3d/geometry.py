from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import cv2
import numpy as np

from .inference import Detection


@dataclass
class ShadowEstimate:
    detection_id: str
    side: str
    shadow_found: bool
    shadow_start_px: int | None = None
    shadow_end_px: int | None = None
    target_ground_range_m: float | None = None
    shadow_end_ground_range_m: float | None = None
    shadow_length_m: float | None = None
    height_m: float | None = None
    quality: float = 0.0
    note: str = ""

    def to_dict(self):
        return asdict(self)


def normalize_percentile(gray: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    x = gray.astype(np.float32)
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def detect_nadir_column(gray: np.ndarray) -> tuple[int, float]:
    """Estimate the central nadir/water-column center from column darkness.

    Returns (column, confidence). It intentionally searches only the central 50%
    so ordinary dark shadows near image edges are less likely to be selected.
    """
    g = normalize_percentile(gray)
    h, w = g.shape
    darkness = 1.0 - np.median(g, axis=0)
    darkness = cv2.GaussianBlur(darkness.reshape(1, -1), (0, 0), sigmaX=max(2, w / 200)).ravel()
    lo, hi = int(w * 0.25), int(w * 0.75)
    segment = darkness[lo:hi]
    if segment.size == 0:
        return w // 2, 0.0
    local_idx = int(np.argmax(segment))
    x = lo + local_idx
    baseline = float(np.median(segment))
    peak = float(segment[local_idx])
    spread = float(np.std(segment)) + 1e-6
    confidence = float(np.clip((peak - baseline) / (3.0 * spread), 0.0, 1.0))
    return x, confidence


def signed_ground_axis(
    width: int,
    nadir_x: int,
    max_range_m: float,
    altitude_m: float | None,
    range_mode: str,
) -> np.ndarray:
    """Map image x columns to signed cross-track ground range.

    range_mode='slant' assumes edge distance maps linearly to slant range and
    converts to ground range with sqrt(r^2-H^2). range_mode='ground' assumes
    the supplied image is already ground-range corrected.
    """
    x = np.arange(width, dtype=np.float64)
    signed_px = x - float(nadir_x)
    denom = np.where(signed_px >= 0, max(width - 1 - nadir_x, 1), max(nadir_x, 1))
    frac = np.abs(signed_px) / denom

    if range_mode.lower().startswith("slant"):
        if altitude_m is None or altitude_m <= 0:
            raise ValueError("Altitude is required for slant-range conversion.")
        slant = frac * float(max_range_m)
        ground = np.sqrt(np.maximum(slant**2 - float(altitude_m) ** 2, 0.0))
    else:
        ground = frac * float(max_range_m)

    return np.sign(signed_px) * ground


def _longest_dark_run(signal: np.ndarray, threshold: float, min_run: int = 3) -> tuple[int, int, float] | None:
    mask = signal < threshold
    best = None
    start = None
    for i, is_dark in enumerate(mask):
        if is_dark and start is None:
            start = i
        if (not is_dark or i == len(mask) - 1) and start is not None:
            end = i if is_dark and i == len(mask) - 1 else i - 1
            length = end - start + 1
            if length >= min_run:
                mean_dark = float(np.mean(signal[start : end + 1]))
                score = length * max(0.0, threshold - mean_dark)
                if best is None or score > best[2]:
                    best = (start, end, score)
            start = None
    return best


def estimate_shadow_height(
    gray: np.ndarray,
    det: Detection,
    nadir_x: int,
    ground_axis_m: np.ndarray,
    altitude_m: float,
    search_factor: float = 5.0,
) -> ShadowEstimate:
    """Heuristic highlight/shadow estimate for an already detected target.

    This is a physics-assisted estimate, not a replacement for manually checked
    shadow picking or MBES. It searches *away from nadir*, where an SSS shadow
    is expected geometrically.
    """
    h, w = gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cx = (x1 + x2) / 2.0
    side = "starboard" if cx >= nadir_x else "port"
    direction = 1 if side == "starboard" else -1

    pad_y = max(2, int((y2 - y1 + 1) * 0.20))
    ya, yb = max(0, y1 - pad_y), min(h, y2 + pad_y + 1)
    box_w = max(4, x2 - x1 + 1)
    outer = x2 if direction > 0 else x1
    max_search = max(10, int(box_w * search_factor))
    end = int(np.clip(outer + direction * max_search, 0, w - 1))

    if direction > 0:
        xs = np.arange(min(outer + 1, w - 1), end + 1)
    else:
        xs = np.arange(max(outer - 1, 0), end - 1, -1)

    if xs.size < 5:
        return ShadowEstimate(det.id, side, False, note="Insufficient outward search area")

    norm = normalize_percentile(gray)
    col_signal = np.median(norm[ya:yb, :][:, xs], axis=0)

    # Local seabed level is estimated outside the target. A shadow should be
    # substantially darker than this, but the threshold is bounded to avoid
    # classifying normal low-backscatter seabed as shadow too easily.
    local_level = float(np.percentile(col_signal, 85))
    dark_threshold = float(np.clip(local_level * 0.58, 0.10, 0.45))
    run = _longest_dark_run(col_signal, dark_threshold, min_run=max(3, box_w // 18))
    if run is None:
        return ShadowEstimate(det.id, side, False, note="No stable outward dark run detected")

    rs, re, score = run
    shadow_start = int(xs[rs])
    shadow_end = int(xs[re])
    target_col = int(np.clip(outer, 0, w - 1))

    g_target = abs(float(ground_axis_m[target_col]))
    g_end = abs(float(ground_axis_m[shadow_end]))
    L = g_end - g_target
    if L <= 0 or g_end <= 0:
        return ShadowEstimate(det.id, side, False, shadow_start, shadow_end, note="Invalid ground-range geometry")

    height_m = float(altitude_m * L / (g_target + L))
    darkness_strength = max(0.0, dark_threshold - float(np.mean(col_signal[rs : re + 1])))
    continuity = min(1.0, (re - rs + 1) / max(5.0, box_w))
    quality = float(np.clip((darkness_strength / max(dark_threshold, 1e-6)) * 0.65 + continuity * 0.35, 0, 1))

    return ShadowEstimate(
        detection_id=det.id,
        side=side,
        shadow_found=True,
        shadow_start_px=shadow_start,
        shadow_end_px=shadow_end,
        target_ground_range_m=g_target,
        shadow_end_ground_range_m=g_end,
        shadow_length_m=float(L),
        height_m=height_m,
        quality=quality,
        note="Automatic shadow pick; review visually before treating as a metric measurement.",
    )


def build_detection_aware_relief(
    gray: np.ndarray,
    detections: Iterable[Detection],
    physical_heights: dict[str, float] | None = None,
    qualitative_height: float = 1.0,
    seabed_texture: float = 0.03,
) -> np.ndarray:
    """Create a detection-aware 3D surface without turning shadows into trenches.

    Only positive local returns/edges are elevated strongly, and that elevation is
    restricted to detector ROIs. Dark acoustic shadows are left near the seabed.
    When a shadow-derived physical height is available, the ROI is scaled so its
    maximum approaches that estimated height.
    """
    physical_heights = physical_heights or {}
    g = normalize_percentile(gray)
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=max(5.0, min(gray.shape) / 35.0))
    positive = np.maximum(g - bg, 0.0)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    if edge.max() > 1e-9:
        edge /= edge.max()

    structure = 0.78 * positive + 0.22 * edge
    p99 = float(np.percentile(structure, 99.5))
    if p99 > 1e-8:
        structure = np.clip(structure / p99, 0, 1)
    structure = np.maximum(structure - 0.08, 0) / 0.92
    structure = cv2.GaussianBlur(structure, (0, 0), 0.8)

    # Small, near-flat seabed texture. It is centered around zero but clipped so
    # dark acoustic shadows cannot become artificial canyons.
    base = cv2.GaussianBlur(g, (0, 0), 2.0) - cv2.GaussianBlur(g, (0, 0), 12.0)
    base = np.clip(base, -0.20, 0.20) * seabed_texture
    z = base.astype(np.float32)

    H, W = gray.shape
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        pad_x = max(2, int((x2 - x1) * 0.15))
        pad_y = max(2, int((y2 - y1) * 0.15))
        xa, xb = max(0, x1 - pad_x), min(W, x2 + pad_x + 1)
        ya, yb = max(0, y1 - pad_y), min(H, y2 + pad_y + 1)

        roi = structure[ya:yb, xa:xb].copy()
        # Soft window avoids a vertical wall at the bounding-box boundary.
        wy = np.hanning(max(3, roi.shape[0]))[: roi.shape[0]]
        wx = np.hanning(max(3, roi.shape[1]))[: roi.shape[1]]
        window = np.outer(wy, wx).astype(np.float32)
        if window.max() > 0:
            window /= window.max()
        roi *= 0.30 + 0.70 * window

        target_height = float(physical_heights.get(det.id, qualitative_height))
        max_roi = float(np.max(roi))
        if max_roi > 1e-8:
            roi = roi / max_roi * target_height
        z[ya:yb, xa:xb] = np.maximum(z[ya:yb, xa:xb], roi)

    return z


def metric_dimensions(
    det: Detection,
    ground_axis_m: np.ndarray | None,
    along_track_m_per_px: float | None,
) -> tuple[float | None, float | None]:
    cross_track = None
    along_track = None
    if ground_axis_m is not None:
        x1 = int(np.clip(round(det.x1), 0, len(ground_axis_m) - 1))
        x2 = int(np.clip(round(det.x2), 0, len(ground_axis_m) - 1))
        cross_track = abs(float(ground_axis_m[x2] - ground_axis_m[x1]))
    if along_track_m_per_px is not None and along_track_m_per_px > 0:
        along_track = abs(det.y2 - det.y1) * float(along_track_m_per_px)
    return along_track, cross_track
