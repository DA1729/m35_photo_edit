from __future__ import annotations

import dataclasses
import math
from typing import Any

import cv2
import numpy as np

from .color_space import luma, rgb_to_lab, srgb_to_linear
from .filters import downscale_for_stats

CAST_HUE = {
    "B": "yellow / green (classic film-base fog)",
    "G": "magenta",
    "R": "cyan / blue-green",
}


@dataclasses.dataclass
class FrameStats:
    path: str
    width: int
    height: int
    percentiles: dict[str, dict[str, float]]
    channel_range: dict[str, float]
    linear_gain: dict[str, float]
    cast_ratio: float
    cast_severity: str
    cast_hue: str
    clipped_low: dict[str, float]
    clipped_high: dict[str, float]
    shadows_clipped: bool
    highlights_clipped: bool
    min_channel_range: float
    effective_bits: float
    detail_score: float
    fog_unevenness: float
    underexposed: bool
    recoverability: str
    preset: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def channel_percentiles(channel8: np.ndarray) -> dict[str, float]:
    q = [0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]
    vals = np.percentile(channel8, q)
    return {f"p{p:g}": float(v) for p, v in zip(q, vals)}


def classify(min_range: float, cast_ratio: float, detail_score: float,
             worst_clip_low: float) -> str:
    if min_range < 34 or detail_score < 1.0 or worst_clip_low > 12.0:
        return "extreme"
    if min_range < 60 or cast_ratio >= 2.7 or worst_clip_low > 2.0:
        return "severe"
    if cast_ratio >= 1.5 or min_range < 130:
        return "moderate"
    return "light"


def severity_of(cast_ratio: float) -> str:
    if cast_ratio < 1.15:
        return "none"
    if cast_ratio < 1.5:
        return "mild"
    if cast_ratio < 2.2:
        return "moderate"
    if cast_ratio < 3.0:
        return "strong"
    return "extreme"


def grade_recoverability(min_range: float, detail_score: float) -> str:
    if min_range >= 90 and detail_score > 3.0:
        return "good"
    if min_range >= 55 and detail_score > 2.0:
        return "fair"
    if min_range >= 32 and detail_score > 1.0:
        return "poor"
    return "minimal"


def analyse(rgb8: np.ndarray, path: str, cfg: dict[str, Any]) -> FrameStats:
    h, w = rgb8.shape[:2]
    small = downscale_for_stats(rgb8)
    smooth = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.0)

    pcts, rng, gains, clip_lo, clip_hi = {}, {}, {}, {}, {}
    lo_p, hi_p = cfg["black_percentile"], cfg["white_percentile"]
    for i, name in enumerate("RGB"):
        ch = smooth[..., i]
        raw = small[..., i].astype(np.float32)
        pcts[name] = channel_percentiles(raw)
        lo, hi = np.percentile(ch, [lo_p, hi_p])
        rng[name] = float(hi - lo)
        lin_hi, lin_lo = srgb_to_linear(np.array([hi / 255.0, lo / 255.0]))
        gains[name] = float(1.0 / max(float(lin_hi - lin_lo), 1e-4))
        clip_lo[name] = float(100.0 * (raw <= 1.0).mean())
        clip_hi[name] = float(100.0 * (raw >= 254.0).mean())

    g = np.array([gains["R"], gains["G"], gains["B"]], dtype=np.float64)
    cast_ratio = float(g.max() / max(g.min(), 1e-6))

    med = np.array([pcts[c]["p50"] for c in "RGB"], dtype=np.float64)
    cast_hue = CAST_HUE["RGB"[int(np.argmin(med))]]
    if cast_ratio < 1.15:
        cast_hue = "neutral"

    severity = severity_of(cast_ratio)
    min_range = float(min(rng.values()))
    effective_bits = float(math.log2(max(min_range, 1.0)))

    lum = luma(small.astype(np.float32) / 255.0)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 2.0)
    detail_score = float(np.std(hp) * 255.0)

    lab_small = rgb_to_lab(small.astype(np.float32) / 255.0)
    sigma = max(4.0, 0.16 * max(lab_small.shape[:2]))
    a_lf = cv2.GaussianBlur(lab_small[..., 1], (0, 0), sigma)
    b_lf = cv2.GaussianBlur(lab_small[..., 2], (0, 0), sigma)
    fog_unevenness = float(math.hypot(float(np.std(a_lf)), float(np.std(b_lf))))

    shadows_clipped = any(v > 0.5 for v in clip_lo.values())
    highlights_clipped = any(v > 0.5 for v in clip_hi.values())
    underexposed = min_range < 55.0 or effective_bits < 5.8
    recoverability = grade_recoverability(min_range, detail_score)
    preset = classify(min_range, cast_ratio, detail_score, max(clip_lo.values()))

    notes: list[str] = []
    if underexposed:
        notes.append(
            f"thin tonal range: worst channel spans only {min_range:.0f}/255 code values "
            f"(~{effective_bits:.1f} bits) -- expect banding/noise after the "
            f"{max(gains.values()):.1f}x stretch"
        )
    if shadows_clipped:
        worst = max(clip_lo, key=clip_lo.get)
        notes.append(
            f"shadow clipping: {clip_lo[worst]:.1f}% of the {worst} channel is crushed to 0 "
            f"(unrecoverable)"
        )
    if highlights_clipped:
        worst = max(clip_hi, key=clip_hi.get)
        notes.append(
            f"highlight clipping: {clip_hi[worst]:.2f}% of the {worst} channel is blown to 255"
        )
    if fog_unevenness > 4.0:
        notes.append(
            f"uneven colour fog / light leak detected "
            f"(low-frequency chroma drift {fog_unevenness:.1f} LAB)"
        )
    if detail_score < 1.2:
        notes.append(
            "very little image structure present -- this frame may be a blank/fogged exposure"
        )
    if recoverability in ("poor", "minimal"):
        notes.append(
            "colour is likely beyond faithful recovery; the _bw output will usually be the "
            "better rescue"
        )

    return FrameStats(
        path=path, width=w, height=h,
        percentiles=pcts, channel_range=rng, linear_gain=gains,
        cast_ratio=cast_ratio, cast_severity=severity, cast_hue=cast_hue,
        clipped_low=clip_lo, clipped_high=clip_hi,
        shadows_clipped=shadows_clipped, highlights_clipped=highlights_clipped,
        min_channel_range=min_range, effective_bits=effective_bits,
        detail_score=detail_score, fog_unevenness=fog_unevenness,
        underexposed=underexposed, recoverability=recoverability,
        preset=preset, notes=notes,
    )
