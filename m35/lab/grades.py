from __future__ import annotations

import numpy as np

from ..color_space import lab_to_rgb, luma, rgb_to_lab


def _curve(x: np.ndarray, pivot: float, strength: float) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    shifted = np.clip(t + (pivot - 0.5) * 0.2, 0.0, 1.0)
    return np.clip((1.0 - strength) * shifted + strength * s, 0.0, 1.0)


def split_tone(gray: np.ndarray, shadow_ab: tuple[float, float],
               highlight_ab: tuple[float, float], pivot: float = 0.5) -> np.ndarray:
    mono = np.repeat(np.clip(gray, 0.0, 1.0)[..., None], 3, axis=2)
    lab = rgb_to_lab(mono)
    t = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    w_hi = np.clip((t - pivot) / max(1.0 - pivot, 1e-3), 0.0, 1.0)
    w_lo = np.clip((pivot - t) / max(pivot, 1e-3), 0.0, 1.0)
    lab[..., 1] = shadow_ab[0] * w_lo + highlight_ab[0] * w_hi
    lab[..., 2] = shadow_ab[1] * w_lo + highlight_ab[1] * w_hi
    return lab_to_rgb(lab)


def selenium(gray: np.ndarray) -> np.ndarray:
    return split_tone(gray, shadow_ab=(4.0, -9.0), highlight_ab=(2.0, 5.0), pivot=0.45)


def sepia(gray: np.ndarray) -> np.ndarray:
    return split_tone(gray, shadow_ab=(6.0, 14.0), highlight_ab=(3.0, 20.0), pivot=0.5)


def cyanotype(gray: np.ndarray) -> np.ndarray:
    return split_tone(gray, shadow_ab=(-2.0, -26.0), highlight_ab=(-8.0, -14.0), pivot=0.5)


def bleach_bypass(rgb: np.ndarray, strength: float = 0.55, lift: float = 0.06) -> np.ndarray:
    lab = rgb_to_lab(rgb)
    lab[..., 1] *= (1.0 - 0.70 * strength)
    lab[..., 2] *= (1.0 - 0.70 * strength)
    base = lab_to_rgb(lab)

    gray = np.clip(luma(base) * 1.15, 0.0, 1.0)
    high = _curve(gray, 0.5, 0.9)[..., None]
    screen = 1.0 - (1.0 - base) * (1.0 - high)
    out = (1.0 - strength) * base + strength * screen
    return np.clip(out * (1.0 - lift) + lift, 0.0, 1.0)


def cross_process(rgb: np.ndarray, strength: float = 0.6) -> np.ndarray:
    out = np.clip(rgb, 0.0, 1.0).copy()
    r = _curve(out[..., 0], 0.55, 0.75)
    g = _curve(out[..., 1], 0.50, 0.55)
    b = np.clip(out[..., 2] * 0.88 + 0.10, 0.0, 1.0)
    shifted = np.stack([r, g, b], axis=2)
    return np.clip((1.0 - strength) * out + strength * shifted, 0.0, 1.0)


GRADES = {
    "bleach": bleach_bypass,
    "cross": cross_process,
}

MONO_GRADES = {
    "selenium": selenium,
    "sepia": sepia,
    "cyanotype": cyanotype,
}
