from __future__ import annotations

import cv2
import numpy as np

from ..color_space import lab_to_rgb, luma, rgb_to_lab, srgb_to_linear, linear_to_srgb

HALATION_TINT = np.array([1.00, 0.42, 0.22], dtype=np.float32)


def halation(rgb: np.ndarray, threshold: float = 0.72, strength: float = 0.16,
             sigma_frac: float = 0.010) -> np.ndarray:
    if strength <= 0.0:
        return rgb
    lin = srgb_to_linear(rgb)
    l = luma(rgb)
    mask = np.clip((l - threshold) / max(1e-3, 1.0 - threshold), 0.0, 1.0) ** 2
    sigma = max(2.0, sigma_frac * max(rgb.shape[:2]))
    bloom = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    glow = bloom[..., None] * HALATION_TINT.reshape(1, 1, 3) * strength
    return linear_to_srgb(np.clip(lin + glow, 0.0, 1.0))


def add_grain(rgb: np.ndarray, amount: float = 0.020, size: float = 0.7,
              seed: int = 1729) -> np.ndarray:
    if amount <= 0.0:
        return rgb
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    noise = rng.standard_normal((h, w)).astype(np.float32)
    if size > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), size)
        noise /= max(float(noise.std()), 1e-6)

    lab = rgb_to_lab(rgb)
    l_norm = lab[..., 0] / 100.0
    weight = (4.0 * l_norm * (1.0 - l_norm)).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] + noise * weight * amount * 100.0, 0.0, 100.0)
    return lab_to_rgb(lab)


def vignette(rgb: np.ndarray, amount: float = 0.18, radius: float = 0.85) -> np.ndarray:
    if amount <= 0.0:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / max(radius, 1e-3)
    fall = 1.0 - amount * np.clip(r - 1.0, 0.0, None) ** 2
    return np.clip(rgb * np.clip(fall, 0.0, 1.0)[..., None], 0.0, 1.0)
