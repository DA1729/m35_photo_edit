from __future__ import annotations

from typing import Any

import numpy as np

from .color_space import lab_to_rgb, rgb_to_lab


def shape_color(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any],
                saturation: float | None = None) -> np.ndarray:
    saturation = cfg["saturation"] if saturation is None else saturation

    lab = rgb_to_lab(rgb)
    a, b = lab[..., 1], lab[..., 2]
    c = np.sqrt(a * a + b * b) + 1e-6
    hue = np.degrees(np.arctan2(b, a)) % 360.0

    yellow_green = np.exp(-0.5 * ((hue - 118.0) / 34.0) ** 2)
    if cfg["skin_protect"]:
        skin = (np.exp(-0.5 * ((hue - 52.0) / 16.0) ** 2)
                * np.exp(-0.5 * ((c - 22.0) / 16.0) ** 2))
        yellow_green = np.clip(yellow_green - 0.85 * skin, 0.0, 1.0)

    damp = 1.0 - (1.0 - cfg["yellow_green_damp"]) * yellow_green
    c_target = c * saturation * damp

    knee, ceiling = cfg["chroma_soft_knee"], cfg["chroma_ceiling"]
    head = max(ceiling - knee, 1.0)
    over = np.maximum(c_target - knee, 0.0)
    c_out = np.where(c_target <= knee, c_target, knee + head * np.tanh(over / head))

    scale = (c_out / c).astype(np.float32)
    lab[..., 1] = a * scale
    lab[..., 2] = b * scale

    params["saturation"] = round(float(saturation), 3)
    params["yellow_green_damp"] = round(float(cfg["yellow_green_damp"]), 3)
    return lab_to_rgb(lab)


def warm_grade(rgb: np.ndarray, cfg: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    da, db = float(cfg["film_warm_a"]), float(cfg["film_warm_b"])
    if da == 0.0 and db == 0.0:
        return rgb
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    taper = (1.0 - np.clip((L - 70.0) / 30.0, 0.0, 1.0) * 0.8).astype(np.float32)
    lab[..., 1] += da * taper
    lab[..., 2] += db * taper
    params["film_warm_ab"] = [da, db]
    return lab_to_rgb(lab)
