from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import analyse
from ..color_space import luma
from ..config import build_config
from ..image_io import load_image, save_jpeg
from ..noise import sharpen
from ..render import to_uint8
from .film_look import add_grain, halation, vignette
from .fusion import LAB_DEFAULTS, render_fused
from .grades import GRADES, MONO_GRADES
from .local_contrast import dehaze

VARIANTS = ("fused", "fused_haze", "bleach", "cross", "selenium", "sepia", "cyanotype")


def finish(rgb: np.ndarray, cfg: dict[str, Any], grain: float, glow: float) -> np.ndarray:
    x = halation(rgb, strength=glow)
    x = sharpen(x, cfg, {}, scale=0.6)
    x = add_grain(x, amount=grain)
    return vignette(x, amount=0.12)


def process_experiment(path: Path, out_dir: Path, base_cfg: dict[str, Any],
                       lab_cfg: dict[str, Any] | None = None,
                       out_stem: str | None = None) -> dict[str, Any]:
    lab_cfg = {**LAB_DEFAULTS, **(lab_cfg or {})}
    rgb, exif, _ = load_image(path)
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    stats = analyse(rgb8, str(path), base_cfg)
    cfg = build_config(base_cfg, stats.preset, {})

    fused, params = render_fused(rgb, stats, base_cfg, lab_cfg)

    grain = float(lab_cfg.get("grain", 0.018))
    glow = float(lab_cfg.get("halation", 0.14))
    quality = cfg["jpeg_quality"]
    dither = cfg["dither"]
    stem = out_stem or path.stem

    outputs: dict[str, np.ndarray] = {}
    outputs["fused"] = finish(fused, cfg, grain, glow)
    outputs["fused_haze"] = finish(
        dehaze(fused, strength=float(lab_cfg.get("dehaze_strength", 0.55))), cfg, grain, glow
    )

    for name, fn in GRADES.items():
        outputs[name] = finish(fn(fused), cfg, grain, glow)

    gray = luma(fused)
    for name, fn in MONO_GRADES.items():
        outputs[name] = finish(fn(gray), cfg, grain, glow * 0.6)

    for name, img in outputs.items():
        save_jpeg(out_dir / f"{stem}_{name}.jpg", to_uint8(img, dither), exif, quality)

    thumbs = {k: v for k, v in outputs.items()}
    return {"stats": stats, "params": params, "outputs": thumbs, "original": rgb8, "stem": stem}
