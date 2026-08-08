from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .analysis import FrameStats
from .color_space import lab_to_rgb, linear_to_srgb, rgb_to_lab, srgb_to_linear
from .filters import downscale_for_stats, guided_filter, soft_shoulder


def predenoise_crushed_channels(rgb: np.ndarray, stats: FrameStats,
                                cfg: dict[str, Any]) -> np.ndarray:
    mode = cfg["predenoise"]
    if mode == "off":
        return rgb

    ranges = np.array([stats.channel_range[c] for c in "RGB"], dtype=np.float32)
    thr = float(cfg["predenoise_range_threshold"])
    out = rgb.copy()

    guide_i = int(np.argmax(ranges))
    guide_range = float(ranges[guide_i])
    use_guide = (cfg["crosschannel_guide"] == "on") or (
        cfg["crosschannel_guide"] == "auto"
        and guide_range > 0.0
        and float(ranges.min()) / guide_range < float(cfg["crosschannel_ratio"])
    )

    if use_guide:
        g = rgb[..., guide_i]
        g_lo, g_hi = float(g.min()), float(g.max())
        guide = (g - g_lo) / max(g_hi - g_lo, 1e-6)
        radius = max(2, int(round(max(rgb.shape[:2]) * float(cfg["guide_radius_frac"]))))
        r_hi = float(cfg["crosschannel_ratio"])
        r_lo = r_hi * float(cfg["crosschannel_full_frac"])
        for i in range(3):
            if i == guide_i:
                continue
            ratio = float(ranges[i]) / guide_range
            crush = float(np.clip((r_hi - ratio) / max(r_hi - r_lo, 1e-4), 0.0, 1.0))
            if crush <= 0.0:
                continue
            filt = guided_filter(guide, out[..., i], radius, float(cfg["guide_eps"]))
            out[..., i] = out[..., i] * (1.0 - crush) + filt * crush
        return out

    for i, name in enumerate("RGB"):
        rng = stats.channel_range[name]
        if mode == "auto" and rng >= thr:
            continue
        crush = float(np.clip((thr - rng) / thr, 0.0, 1.0))
        sigma_color = cfg["predenoise_sigma_codes"] * (0.55 + 0.9 * crush) / 255.0
        d = 5 if crush < 0.6 else 7
        out[..., i] = cv2.bilateralFilter(
            np.ascontiguousarray(out[..., i]), d, float(sigma_color), 2.0 + 2.0 * crush
        )
    return out


def channel_levels(rgb: np.ndarray, stats: FrameStats, cfg: dict[str, Any],
                   neutrality: float, params: dict[str, Any]) -> np.ndarray:
    lin = srgb_to_linear(rgb)
    small = cv2.GaussianBlur(downscale_for_stats(lin), (0, 0), 1.0)

    lo_p, hi_p = cfg["black_percentile"], cfg["white_percentile"]
    blacks, whites = np.zeros(3, np.float32), np.zeros(3, np.float32)
    for i in range(3):
        lo, hi = np.percentile(small[..., i], [lo_p, hi_p])
        blacks[i] = lo * cfg["black_removal"]
        whites[i] = max(float(hi), float(lo) + 1e-3)

    gains = 1.0 / np.maximum(whites - blacks, 1e-4)
    gmean = float(np.exp(np.mean(np.log(gains))))
    gains_blended = gmean * (gains / gmean) ** neutrality
    bmean = float(np.mean(blacks))
    blacks_blended = bmean + (blacks - bmean) * neutrality

    out = (lin - blacks_blended.reshape(1, 1, 3)) * gains_blended.reshape(1, 1, 3)
    out = np.clip(out, 0.0, 1.0)

    mm = float(cfg["midtone_match"]) * neutrality
    gammas = np.ones(3, np.float32)
    if mm > 0.0:
        stat = downscale_for_stats(out)
        med = np.clip(np.array([np.median(stat[..., i]) for i in range(3)], dtype=np.float64),
                      0.02, 0.98)
        target = float(np.exp(np.mean(np.log(med))))
        g_lo, g_hi = cfg["midtone_gamma_limit"]
        for i in range(3):
            g = math.log(target) / math.log(float(med[i]))
            gammas[i] = float(np.clip(1.0 + (g - 1.0) * mm, g_lo, g_hi))
            out[..., i] = np.power(out[..., i], gammas[i])

    out = np.clip(soft_shoulder(out, cfg["highlight_knee"]), 0.0, 1.0)

    params["levels_black_linear"] = [round(float(v), 5) for v in blacks_blended]
    params["levels_white_linear"] = [round(float(v), 5) for v in whites]
    params["levels_gain_linear"] = [round(float(v), 4) for v in gains_blended]
    params["levels_gain_raw"] = [round(float(v), 4) for v in gains]
    params["midtone_gamma"] = [round(float(v), 4) for v in gammas]
    params["neutrality"] = round(float(neutrality), 3)
    return linear_to_srgb(out)


def remove_residual_cast(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any],
                         strength: float | None = None, warmth: float = 0.0) -> np.ndarray:
    strength = cfg["lab_cast_strength"] if strength is None else strength
    if strength <= 0.0:
        return rgb

    lab = rgb_to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    mid = (L > 18.0) & (L < 88.0)
    if mid.sum() < 64:
        mid = np.ones_like(L, dtype=bool)
    a_bias = float(np.median(a[mid]))
    b_bias = float(np.median(b[mid]))

    cap = cfg["lab_cast_max"]
    da = float(np.clip(a_bias * strength, -cap, cap)) - warmth * 3.0
    db = float(np.clip(b_bias * strength, -cap, cap)) - warmth * 9.0

    hi_taper = 1.0 - np.clip((L - 82.0) / 18.0, 0.0, 1.0) * 0.7
    lo_taper = np.clip(L / 12.0, 0.0, 1.0)
    taper = (hi_taper * lo_taper).astype(np.float32)

    lab[..., 1] = a - da * taper
    lab[..., 2] = b - db * taper

    params["lab_cast_bias_ab"] = [round(a_bias, 2), round(b_bias, 2)]
    params["lab_cast_applied_ab"] = [round(da, 2), round(db, 2)]
    return lab_to_rgb(lab)


def flatten_uneven_fog(rgb: np.ndarray, stats: FrameStats, cfg: dict[str, Any],
                       params: dict[str, Any]) -> np.ndarray:
    mode = cfg["fog_flatten"]
    if mode == "off" or (mode == "auto" and stats.fog_unevenness < 4.0):
        params["fog_flatten"] = "skipped"
        return rgb

    lab = rgb_to_lab(rgb)
    strength = float(cfg["fog_flatten_strength"])
    cap = float(np.clip(cfg["fog_flatten_cap_per_unit"] * stats.fog_unevenness,
                        cfg["fog_flatten_min"], cfg["fog_flatten_max"]))
    guide = np.ascontiguousarray(lab[..., 0] / 100.0)
    radius = max(8, int(round(max(rgb.shape[:2]) * float(cfg["fog_flatten_radius_frac"]))))
    eps = float(cfg["fog_flatten_eps"])

    for idx in (1, 2):
        plane = np.ascontiguousarray(lab[..., idx])
        lf = guided_filter(guide, plane, radius, eps)
        drift = lf - float(np.median(lf))
        lab[..., idx] = plane - np.clip(drift * strength, -cap, cap)

    params["fog_flatten"] = (
        f"applied (guided, r={radius}px, strength={strength}, cap={cap:.0f} LAB)"
    )
    return lab_to_rgb(lab)


def protect_highlights(rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    start, amount = cfg["highlight_desat_start"], cfg["highlight_desat_amount"]
    if amount <= 0.0:
        return rgb
    lab = rgb_to_lab(rgb)
    L = lab[..., 0] / 100.0
    w = np.clip((L - start) / max(1e-3, 1.0 - start), 0.0, 1.0) ** 1.5
    scale = (1.0 - amount * w).astype(np.float32)
    lab[..., 1] *= scale
    lab[..., 2] *= scale
    return lab_to_rgb(lab)
