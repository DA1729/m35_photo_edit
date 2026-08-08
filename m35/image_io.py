from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import SUPPORTED_EXT


def load_image(path: Path) -> tuple[np.ndarray, bytes | None, dict]:
    with Image.open(path) as im:
        exif = im.info.get("exif")
        info = {"mode": im.mode, "size": im.size, "format": im.format}
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im).astype(np.float32)
            arr = arr / max(float(arr.max()), 1.0)
            rgb = np.repeat(arr[..., None], 3, axis=2)
        else:
            rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb), exif, info


def save_jpeg(path: Path, rgb8: np.ndarray, exif: bytes | None, quality: int) -> None:
    img = Image.fromarray(rgb8, mode="RGB")
    kwargs: dict[str, Any] = {"quality": int(quality), "subsampling": 0, "optimize": True}
    if exif:
        try:
            img.save(path, format="JPEG", exif=exif, **kwargs)
            return
        except Exception:
            pass
    img.save(path, format="JPEG", **kwargs)


def collect_images(root: Path, recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)


def resolve_output_stems(files: list[Path]) -> dict[Path, str]:
    by_stem: dict[str, list[Path]] = {}
    for f in files:
        by_stem.setdefault(f.stem, []).append(f)

    stems: dict[Path, str] = {}
    used: set[str] = set()
    for stem, group in by_stem.items():
        if len(group) == 1:
            stems[group[0]] = stem
            used.add(stem)
            continue
        for f in group:
            cand = f"{stem}_{f.suffix.lstrip('.').lower()}"
            n = 2
            while cand in used:
                cand = f"{stem}_{f.suffix.lstrip('.').lower()}_{n}"
                n += 1
            stems[f] = cand
            used.add(cand)
    return stems
