from __future__ import annotations

import cv2
import numpy as np


def build_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [x]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_local_contrast(plane: np.ndarray, levels: int = 5, gain: float = 1.35,
                             clamp: float = 12.0, coarse_gain: float = 1.0) -> np.ndarray:
    pyr = build_pyramid(plane.astype(np.float32), levels)
    out = pyr[-1] * coarse_gain
    for i in range(len(pyr) - 2, -1, -1):
        size = (pyr[i].shape[1], pyr[i].shape[0])
        detail = pyr[i] - cv2.pyrUp(pyr[i + 1], dstsize=size)
        scaled = np.clip(detail * gain, -clamp, clamp)
        out = cv2.pyrUp(out, dstsize=size) + scaled
    return out.astype(np.float32)


def dark_channel(rgb: np.ndarray, patch: int) -> np.ndarray:
    min_rgb = rgb.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_rgb, kernel)


def estimate_airlight(rgb: np.ndarray, dark: np.ndarray, top_frac: float = 0.001) -> np.ndarray:
    flat_dark = dark.ravel()
    n = max(1, int(flat_dark.size * top_frac))
    idx = np.argpartition(flat_dark, -n)[-n:]
    pixels = rgb.reshape(-1, 3)[idx]
    return np.percentile(pixels, 90, axis=0).astype(np.float32)


def dehaze(rgb: np.ndarray, strength: float = 0.7, patch_frac: float = 0.02,
           t_floor: float = 0.25, refine_radius_frac: float = 0.02) -> np.ndarray:
    from ..filters import guided_filter

    h, w = rgb.shape[:2]
    patch = max(3, int(round(min(h, w) * patch_frac)) | 1)
    dark = dark_channel(rgb, patch)
    air = np.maximum(estimate_airlight(rgb, dark, 0.001), 1e-3)

    normalised = rgb / air.reshape(1, 1, 3)
    t = 1.0 - strength * dark_channel(normalised, patch)

    guide = np.ascontiguousarray(rgb.mean(axis=2))
    radius = max(4, int(round(max(h, w) * refine_radius_frac)))
    t = guided_filter(guide, np.ascontiguousarray(t.astype(np.float32)), radius, 1e-3)
    t = np.clip(t, t_floor, 1.0)[..., None]

    out = (rgb - air.reshape(1, 1, 3)) / t + air.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
