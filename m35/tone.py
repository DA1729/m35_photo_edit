from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .color_space import lab_to_rgb, luma, rgb_to_lab
from .filters import downscale_for_stats, soft_shoulder


def film_scurve(x: np.ndarray, amount: float, pivot: float) -> np.ndarray:
    if amount <= 0.0:
        return x
    t = np.clip(x, 0.0, 1.0)
    p = np.clip(pivot, 0.15, 0.85)
    below = t / p
    above = (t - p) / (1.0 - p)
    s_below = p * (below * below * (3.0 - 2.0 * below))
    s_above = p + (1.0 - p) * (above * above * (3.0 - 2.0 * above))
    s = np.where(t < p, s_below, s_above).astype(np.float32)
    return ((1.0 - amount) * t + amount * s).astype(np.float32)


def soft_range(x: np.ndarray, eps: float = 0.055) -> np.ndarray:
    x = 0.5 * (x + np.sqrt(x * x + eps * eps))
    inv = 1.0 - x
    x = 1.0 - 0.5 * (inv + np.sqrt(inv * inv + eps * eps))
    return np.clip(x, 0.0, 1.0)


def tonal_recovery(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any],
                   scurve_amount: float | None = None, lift: float = 0.0) -> np.ndarray:
    scurve_amount = cfg["scurve_amount"] if scurve_amount is None else scurve_amount

    lum = luma(rgb)
    small = cv2.GaussianBlur(downscale_for_stats(lum), (0, 0), 1.0)
    lo = float(np.percentile(small, cfg["tone_black_percentile"]))
    hi = float(np.percentile(small, cfg["tone_white_percentile"]))
    lo = max(0.0, lo - cfg["tone_black_headroom"])
    hi = min(1.0, hi + cfg["tone_white_headroom"])
    if hi - lo < 0.05:
        hi = min(1.0, lo + 0.05)

    x = soft_range((lum - lo) / (hi - lo))
    y = film_scurve(x, scurve_amount, cfg["scurve_pivot"])

    sl = cfg["shadow_lift"] + lift
    if sl > 0.0:
        y = y * (1.0 - sl) + sl * (1.0 - np.power(1.0 - y, 2.2))

    target = cfg["target_median"]
    if target is None:
        target = float(np.clip(0.44 + 0.10 * (float(np.median(x)) - 0.44), 0.36, 0.56))
    cur = float(np.median(y))
    gamma = 1.0
    if 0.02 < cur < 0.98 and target > 0:
        gamma = float(np.clip(math.log(max(target, 1e-4)) / math.log(max(cur, 1e-4)), 0.72, 1.38))
        y = np.power(y, gamma)

    ratio = np.minimum(y / np.maximum(lum, 1e-4), 40.0)
    out = soft_shoulder(rgb * ratio[..., None], 0.94)

    params["tone_black_point"] = round(lo, 4)
    params["tone_white_point"] = round(hi, 4)
    params["tone_scurve"] = round(float(scurve_amount), 3)
    params["tone_gamma"] = round(gamma, 3)
    params["tone_target_median"] = round(float(target), 3)
    return np.clip(out, 0.0, 1.0)


def clahe_luminance(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any],
                    clip: float | None = None, blend: float | None = None) -> np.ndarray:
    clip = cfg["clahe_clip"] if clip is None else clip
    blend = cfg["clahe_blend"] if blend is None else blend
    if clip <= 0.0 or blend <= 0.0:
        return rgb

    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    grid = int(cfg["clahe_grid"])
    eq = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid)).apply(L8)
    L_eq = eq.astype(np.float32) / 2.55

    w = blend * (1.0 - 0.55 * np.clip(np.abs(L - 50.0) / 50.0, 0.0, 1.0) ** 2)
    lab[..., 0] = L * (1.0 - w) + L_eq * w

    params["clahe_clip"] = round(float(clip), 2)
    params["clahe_blend"] = round(float(blend), 2)
    return lab_to_rgb(lab)
