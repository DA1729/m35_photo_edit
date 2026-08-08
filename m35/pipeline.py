from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analysis import analyse
from .config import build_config
from .image_io import load_image, save_jpeg
from .render import render_bw, render_film, render_neutral, to_uint8
from .report import format_report


def thumbnail(a: np.ndarray, width: int) -> np.ndarray:
    h = max(1, int(round(a.shape[0] * width / a.shape[1])))
    return cv2.resize(a, (width, h), interpolation=cv2.INTER_AREA)


def process_file(path: Path, out_dir: Path, base_cfg: dict[str, Any],
                 overrides: dict[str, Any], want_sheet: bool,
                 out_stem: str | None = None) -> dict[str, Any]:
    rgb, exif, _ = load_image(path)
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    stats = analyse(rgb8, str(path), base_cfg)
    if overrides.get("force_preset"):
        stats.preset = overrides["force_preset"]
    cfg = build_config(base_cfg, stats.preset,
                       {k: v for k, v in overrides.items()
                        if k != "force_preset" and v is not None})

    neutral, p_neutral = render_neutral(rgb, stats, cfg)
    film, p_film = render_film(rgb, stats, cfg)
    bw, p_bw = render_bw(rgb, stats, cfg)

    dither = cfg["dither"]
    neutral8 = to_uint8(neutral, dither)
    film8 = to_uint8(film, dither)
    bw8 = to_uint8(bw, dither)

    stem = out_stem or path.stem
    quality = cfg["jpeg_quality"]
    save_jpeg(out_dir / f"{stem}_neutral.jpg", neutral8, exif, quality)
    save_jpeg(out_dir / f"{stem}_film.jpg", film8, exif, quality)
    save_jpeg(out_dir / f"{stem}_bw.jpg", bw8, exif, quality)

    params = {"neutral": p_neutral, "film": p_film, "bw": p_bw}

    sheet_row = None
    if want_sheet:
        tw = base_cfg["contact_thumb_width"]
        sheet_row = {
            "name": path.name,
            "preset": stats.preset,
            "cast": stats.cast_severity,
            "cast_ratio": stats.cast_ratio,
            "min_range": stats.min_channel_range,
            "recoverability": stats.recoverability,
            "original": thumbnail(rgb8, tw),
            "neutral": thumbnail(neutral8, tw),
            "film": thumbnail(film8, tw),
            "bw": thumbnail(bw8, tw),
        }

    return {
        "stats": stats,
        "params": params,
        "report": format_report(stats, params),
        "sheet_row": sheet_row,
    }
