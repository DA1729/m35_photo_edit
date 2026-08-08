from __future__ import annotations

from typing import Any

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DEFAULTS: dict[str, Any] = {
    "black_percentile": 0.60,
    "white_percentile": 99.60,
    "black_removal": 1.0,
    "neutrality": 0.93,
    "midtone_match": 0.88,
    "midtone_gamma_limit": (0.45, 2.30),

    "highlight_knee": 0.72,
    "highlight_desat_start": 0.82,
    "highlight_desat_amount": 0.75,

    "lab_cast_strength": 0.45,
    "lab_cast_max": 30.0,

    "fog_flatten": "auto",
    "fog_flatten_strength": 0.85,
    "fog_flatten_min": 10.0,
    "fog_flatten_max": 46.0,
    "fog_flatten_cap_per_unit": 3.2,
    "fog_flatten_radius_frac": 0.06,
    "fog_flatten_eps": 2e-3,

    "tone_black_percentile": 0.35,
    "tone_white_percentile": 99.75,
    "tone_black_headroom": 0.006,
    "tone_white_headroom": 0.004,
    "scurve_amount": 0.34,
    "scurve_pivot": 0.46,
    "shadow_lift": 0.05,
    "target_median": None,

    "clahe_clip": 1.5,
    "clahe_grid": 8,
    "clahe_blend": 0.60,

    "saturation": 1.06,
    "chroma_soft_knee": 34.0,
    "chroma_ceiling": 62.0,
    "yellow_green_damp": 0.80,
    "skin_protect": True,

    "predenoise": "auto",
    "predenoise_range_threshold": 110.0,
    "predenoise_sigma_codes": 2.6,
    "crosschannel_guide": "auto",
    "crosschannel_ratio": 0.62,
    "crosschannel_full_frac": 0.45,
    "guide_radius_frac": 0.0032,
    "guide_eps": 4e-4,
    "chroma_denoise": 1.0,
    "chroma_denoise_scale": 4,
    "luma_denoise": 0.0,
    "chroma_confidence": 0.85,
    "chroma_confidence_floor": 0.22,
    "chroma_confidence_lo_pct": 20.0,
    "chroma_confidence_hi_pct": 80.0,
    "grain_keep": 0.30,
    "dither": 0.55,

    "sharpen_amount": 0.32,
    "sharpen_sigma": 1.05,
    "sharpen_threshold": 0.010,
    "sharpen_clamp": 0.020,

    "film_warm_a": 1.8,
    "film_warm_b": 6.5,
    "film_neutrality_scale": 0.98,
    "film_softness": 0.55,
    "film_saturation": 0.94,
    "film_lift": 0.020,

    "bw_snr_weighting": 0.75,
    "bw_scurve_amount": 0.46,
    "bw_clahe_clip": 2.0,
    "bw_clahe_blend": 0.72,
    "bw_sharpen_scale": 1.25,

    "jpeg_quality": 95,
    "contact_thumb_width": 520,
}

PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "neutrality": 0.62,
        "midtone_match": 0.40,
        "chroma_confidence": 0.30,
        "lab_cast_strength": 0.18,
        "lab_cast_max": 8.0,
        "scurve_amount": 0.18,
        "clahe_clip": 1.2,
        "clahe_blend": 0.45,
        "chroma_denoise": 0.7,
        "predenoise": "off",
    },
    "moderate": {},
    "severe": {
        "neutrality": 0.95,
        "midtone_match": 0.90,
        "chroma_confidence": 0.90,
        "chroma_denoise_scale": 5,
        "saturation": 0.86,
        "lab_cast_strength": 0.50,
        "highlight_knee": 0.68,
        "scurve_amount": 0.38,
        "shadow_lift": 0.07,
        "clahe_clip": 1.7,
        "clahe_blend": 0.65,
        "predenoise_sigma_codes": 3.4,
        "chroma_denoise": 1.6,
        "luma_denoise": 0.0,
        "sharpen_amount": 0.26,
        "dither": 0.85,
        "yellow_green_damp": 0.74,
    },
    "extreme": {
        "neutrality": 0.97,
        "midtone_match": 0.92,
        "chroma_confidence": 0.95,
        "chroma_confidence_floor": 0.12,
        "chroma_denoise_scale": 6,
        "lab_cast_strength": 0.55,
        "highlight_knee": 0.62,
        "highlight_desat_start": 0.76,
        "scurve_amount": 0.30,
        "shadow_lift": 0.09,
        "clahe_clip": 1.9,
        "clahe_blend": 0.70,
        "predenoise_sigma_codes": 4.2,
        "chroma_denoise": 2.2,
        "luma_denoise": 0.15,
        "saturation": 0.72,
        "sharpen_amount": 0.20,
        "dither": 1.1,
        "yellow_green_damp": 0.68,
    },
}


def build_config(base: dict[str, Any], preset: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)
    cfg.update(PRESETS.get(preset, {}))
    cfg.update(overrides)
    return cfg
