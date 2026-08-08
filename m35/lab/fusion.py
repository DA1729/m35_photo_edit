from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis import FrameStats
from ..color_cast import (channel_levels, flatten_uneven_fog, predenoise_crushed_channels,
                          protect_highlights, remove_residual_cast)
from ..color_space import lab_to_rgb, rgb_to_lab
from ..config import build_config
from ..noise import chroma_confidence_map, denoise
from ..render import bw_channel_weights
from ..tone import tonal_recovery
from .local_contrast import laplacian_local_contrast

LAB_DEFAULTS: dict[str, Any] = {
    "chroma_gain": 3.5,
    "chroma_denoise": 3.0,
    "chroma_denoise_scale": 8,
    "lc_levels": 5,
    "lc_gain": 1.35,
    "lc_clamp": 12.0,
    "chroma_ceiling_lab": 38.0,
    "gain_floor": 0.7,
    "gain_lo_pct": 25.0,
    "gain_hi_pct": 85.0,
}


def corrected_base(rgb: np.ndarray, stats: FrameStats, cfg: dict[str, Any],
                   params: dict[str, Any]) -> np.ndarray:
    x = predenoise_crushed_channels(rgb, stats, cfg)
    x = channel_levels(x, stats, cfg, cfg["neutrality"], params)
    x = flatten_uneven_fog(x, stats, cfg, params)
    x = remove_residual_cast(x, cfg, params)
    return protect_highlights(x, cfg)


def information_luma(base: np.ndarray, stats: FrameStats, cfg: dict[str, Any]) -> np.ndarray:
    w = bw_channel_weights(stats, cfg)
    toned = tonal_recovery(base, cfg, {}, scurve_amount=cfg["bw_scurve_amount"])
    gray = np.clip((toned * w.reshape(1, 1, 3)).sum(axis=2), 0.0, 1.0)
    return rgb_to_lab(np.repeat(gray[..., None], 3, axis=2))[..., 0]


def render_fused(rgb: np.ndarray, stats: FrameStats, base_cfg: dict[str, Any],
                 lab_cfg: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
    lab_cfg = {**LAB_DEFAULTS, **(lab_cfg or {})}
    p: dict[str, Any] = {"variant": "fused"}

    cfg = build_config(base_cfg, stats.preset, {
        "chroma_confidence": 0.0,
        "chroma_denoise": lab_cfg["chroma_denoise"],
        "chroma_denoise_scale": lab_cfg["chroma_denoise_scale"],
    })

    base = corrected_base(rgb, stats, cfg, p)

    color = tonal_recovery(base, cfg, p)
    color = denoise(color, cfg, p)
    lab = rgb_to_lab(color)

    luma_plane = information_luma(base, stats, cfg)
    luma_plane = laplacian_local_contrast(
        luma_plane,
        levels=int(lab_cfg["lc_levels"]),
        gain=float(lab_cfg["lc_gain"]),
        clamp=float(lab_cfg["lc_clamp"]),
    )

    gain = float(lab_cfg["chroma_gain"])
    conf = chroma_confidence_map(lab[..., 0], max(rgb.shape[:2]))
    lo_a, hi_a = np.percentile(conf, [float(lab_cfg["gain_lo_pct"]),
                                      float(lab_cfg["gain_hi_pct"])])
    weight = np.clip((conf - lo_a) / max(float(hi_a - lo_a), 1e-4), 0.0, 1.0)
    floor = float(lab_cfg["gain_floor"])
    gain_map = (floor + (gain - floor) * weight).astype(np.float32)

    a = lab[..., 1] * gain_map
    b = lab[..., 2] * gain_map

    ceiling = float(lab_cfg["chroma_ceiling_lab"])
    c = np.sqrt(a * a + b * b) + 1e-6
    limited = ceiling * np.tanh(c / ceiling)
    scale = (limited / c).astype(np.float32)

    lab[..., 0] = np.clip(luma_plane, 0.0, 100.0)
    lab[..., 1] = a * scale
    lab[..., 2] = b * scale

    p["chroma_gain"] = gain
    p["chroma_gain_mean"] = round(float(gain_map.mean()), 3)
    p["lc_gain"] = float(lab_cfg["lc_gain"])
    return lab_to_rgb(lab), p
