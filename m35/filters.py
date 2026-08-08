from __future__ import annotations

import cv2
import numpy as np


def downscale_for_stats(img: np.ndarray, long_side: int = 1100) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def box(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.boxFilter(x, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    k = (2 * radius + 1, 2 * radius + 1)
    mean_g, mean_s = box(guide, k), box(src, k)
    cov = box(guide * src, k) - mean_g * mean_s
    var = box(guide * guide, k) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return (box(a, k) * guide + box(b, k)).astype(np.float32)


def local_std(x: np.ndarray, win: int) -> np.ndarray:
    k = (win, win)
    mean = box(x, k)
    mean_sq = box(x * x, k)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def soft_shoulder(x: np.ndarray, knee: float) -> np.ndarray:
    if knee >= 0.999:
        return np.clip(x, 0.0, None)
    head = 1.0 - knee
    over = np.maximum(x - knee, 0.0)
    return np.where(x <= knee, x, knee + head * np.tanh(over / head)).astype(np.float32)
