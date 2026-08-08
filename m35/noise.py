from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .color_space import lab_to_rgb, rgb_to_lab
from .filters import local_std


def chroma_confidence_map(L: np.ndarray, long_side: int) -> np.ndarray:
    win = max(9, (int(round(long_side / 55.0)) | 1))
    small = cv2.GaussianBlur(L, (0, 0), 1.4)
    large = cv2.GaussianBlur(L, (0, 0), 11.0)
    noise = local_std(L - small, win)
    structure = local_std(small - large, win)
    conf = structure / (structure + 1.6 * noise + 1e-4)
    return cv2.GaussianBlur(conf, (0, 0), max(3.0, long_side / 220.0)).astype(np.float32)


def denoise(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any],
            chroma_scale: float | None = None) -> np.ndarray:
    chroma_scale = cfg["chroma_denoise"] if chroma_scale is None else chroma_scale
    ld = float(cfg["luma_denoise"])
    conf_strength = float(cfg["chroma_confidence"])
    if chroma_scale <= 0.0 and ld <= 0.0 and conf_strength <= 0.0:
        return rgb

    lab = rgb_to_lab(rgb)
    h, w = rgb.shape[:2]
    long_side = max(h, w)

    if chroma_scale > 0.0:
        n = max(1, int(cfg["chroma_denoise_scale"]))
        sw, sh = max(8, w // n), max(8, h // n)
        d = int(max(5, round(7 * min(chroma_scale, 2.0))))
        d += 1 - (d % 2)
        for idx in (1, 2):
            plane = np.ascontiguousarray(lab[..., idx])
            reduced = cv2.resize(plane, (sw, sh), interpolation=cv2.INTER_AREA)
            reduced = cv2.bilateralFilter(reduced, d, 10.0 * chroma_scale, 6.0)
            lab[..., idx] = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_LINEAR)

    if conf_strength > 0.0:
        conf = chroma_confidence_map(lab[..., 0], long_side)
        floor = float(cfg["chroma_confidence_floor"])
        lo_a, hi_a = np.percentile(conf, [float(cfg["chroma_confidence_lo_pct"]),
                                          float(cfg["chroma_confidence_hi_pct"])])
        gate = np.clip((conf - lo_a) / max(float(hi_a - lo_a), 1e-4), 0.0, 1.0)
        scale = floor + (1.0 - floor) * gate
        scale = 1.0 - conf_strength * (1.0 - scale)
        lab[..., 1] *= scale
        lab[..., 2] *= scale
        params["chroma_confidence_mean"] = round(float(scale.mean()), 3)

    if ld > 0.0:
        L = lab[..., 0]
        L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
        h_nlm = float(np.clip(12.0 * ld, 1.0, 14.0))
        smoothed = cv2.fastNlMeansDenoising(L8, None, h_nlm, 7, 21).astype(np.float32) / 2.55
        lab[..., 0] = smoothed + (L - smoothed) * float(cfg["grain_keep"])

    params["chroma_denoise"] = round(float(chroma_scale), 2)
    params["luma_denoise"] = round(ld, 2)
    return lab_to_rgb(lab)


def sharpen(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any],
            scale: float = 1.0) -> np.ndarray:
    amount = cfg["sharpen_amount"] * scale
    if amount <= 0.0:
        return rgb

    lab = rgb_to_lab(rgb)
    L = lab[..., 0] / 100.0
    detail = L - cv2.GaussianBlur(L, (0, 0), cfg["sharpen_sigma"])

    thr = cfg["sharpen_threshold"]
    gate = np.clip((np.abs(detail) - thr) / max(thr, 1e-4), 0.0, 1.0)
    clamp = cfg["sharpen_clamp"]
    lab[..., 0] = np.clip((L + np.clip(detail * gate * amount, -clamp, clamp)) * 100.0, 0.0, 100.0)

    params["sharpen_amount"] = round(float(amount), 3)
    return lab_to_rgb(lab)
