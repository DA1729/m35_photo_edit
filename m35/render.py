from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import FrameStats
from .color_cast import (channel_levels, flatten_uneven_fog, predenoise_crushed_channels,
                         protect_highlights, remove_residual_cast)
from .color_grade import shape_color, warm_grade
from .color_space import luma
from .noise import denoise, sharpen
from .tone import clahe_luminance, tonal_recovery


def to_uint8(rgb: np.ndarray, dither: float) -> np.ndarray:
    x = np.clip(rgb, 0.0, 1.0) * 255.0
    if dither > 0.0:
        rng = np.random.default_rng(12345)
        x = x + rng.random(x.shape, dtype=np.float32) * dither - dither * 0.5
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def render_neutral(rgb: np.ndarray, stats: FrameStats,
                   cfg: dict[str, Any]) -> tuple[np.ndarray, dict]:
    p: dict[str, Any] = {"variant": "neutral"}
    x = predenoise_crushed_channels(rgb, stats, cfg)
    x = channel_levels(x, stats, cfg, cfg["neutrality"], p)
    x = flatten_uneven_fog(x, stats, cfg, p)
    x = remove_residual_cast(x, cfg, p)
    x = protect_highlights(x, cfg)
    x = tonal_recovery(x, cfg, p)
    x = clahe_luminance(x, cfg, p)
    x = shape_color(x, cfg, p)
    x = denoise(x, cfg, p)
    x = sharpen(x, cfg, p)
    return x, p


def render_film(rgb: np.ndarray, stats: FrameStats,
                cfg: dict[str, Any]) -> tuple[np.ndarray, dict]:
    p: dict[str, Any] = {"variant": "film"}
    x = predenoise_crushed_channels(rgb, stats, cfg)
    x = channel_levels(x, stats, cfg, cfg["neutrality"] * cfg["film_neutrality_scale"], p)
    x = flatten_uneven_fog(x, stats, cfg, p)
    x = remove_residual_cast(x, cfg, p, strength=cfg["lab_cast_strength"])
    x = protect_highlights(x, cfg)
    x = tonal_recovery(x, cfg, p,
                       scurve_amount=cfg["scurve_amount"] * 0.85,
                       lift=cfg["film_lift"])
    x = clahe_luminance(x, cfg, p, clip=cfg["clahe_clip"] * 0.8, blend=cfg["clahe_blend"] * 0.8)
    x = shape_color(x, cfg, p, saturation=cfg["saturation"] * cfg["film_saturation"])
    x = denoise(x, cfg, p, chroma_scale=cfg["chroma_denoise"] * 0.85)
    x = warm_grade(x, cfg, p)
    x = sharpen(x, cfg, p, scale=cfg["film_softness"])
    return x, p


def bw_channel_weights(stats: FrameStats, cfg: dict[str, Any]) -> np.ndarray:
    ranges = np.array([stats.channel_range[c] for c in "RGB"], dtype=np.float32)
    info_w = ranges / max(float(ranges.sum()), 1e-6)
    luma_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    k = float(np.clip(cfg["bw_snr_weighting"], 0.0, 1.0))
    w = (1.0 - k) * luma_w + k * info_w
    return w / float(w.sum())


def render_bw(rgb: np.ndarray, stats: FrameStats,
              cfg: dict[str, Any]) -> tuple[np.ndarray, dict]:
    p: dict[str, Any] = {"variant": "bw"}
    x = predenoise_crushed_channels(rgb, stats, cfg)
    x = channel_levels(x, stats, cfg, cfg["neutrality"], p)
    x = flatten_uneven_fog(x, stats, cfg, p)
    x = tonal_recovery(x, cfg, p, scurve_amount=cfg["bw_scurve_amount"])

    w = bw_channel_weights(stats, cfg)
    gray = np.clip((x * w.reshape(1, 1, 3)).sum(axis=2), 0.0, 1.0)
    mono = np.repeat(gray[..., None], 3, axis=2)

    mono = clahe_luminance(mono, cfg, p, clip=cfg["bw_clahe_clip"], blend=cfg["bw_clahe_blend"])
    mono = denoise(mono, cfg, p, chroma_scale=0.0)
    mono = sharpen(mono, cfg, p, scale=cfg["bw_sharpen_scale"])
    mono = np.repeat(luma(mono)[..., None], 3, axis=2)

    p["bw_channel_weights"] = [round(float(v), 4) for v in w]
    return mono, p
