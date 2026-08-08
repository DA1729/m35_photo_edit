#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from m35 import DEFAULTS, SUPPORTED_EXT, collect_images, resolve_output_stems
from m35.contact_sheet import build_contact_sheet
from m35.lab import LAB_DEFAULTS, VARIANTS, process_experiment

SHEET_COLUMNS = ("original", "fused", "bleach", "cyanotype")
SHEET_TITLES = ("ORIGINAL", "FUSED", "BLEACH BYPASS", "CYANOTYPE")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="experiment.py",
        description="Unconventional looks built on the recovered frames. Writes alongside, "
                    "never over, the restore_film.py output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, nargs="?", default=Path("./experiments"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--chroma-gain", type=float, default=None)
    ap.add_argument("--lc-gain", type=float, default=None)
    ap.add_argument("--grain", type=float, default=None)
    ap.add_argument("--halation", type=float, default=None)
    ap.add_argument("--dehaze-strength", type=float, default=None)
    ap.add_argument("--no-contact-sheet", action="store_true")
    return ap.parse_args(argv)


def thumb(a: np.ndarray, width: int) -> np.ndarray:
    h = max(1, int(round(a.shape[0] * width / a.shape[1])))
    return cv2.resize(a, (width, h), interpolation=cv2.INTER_AREA)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.input.is_dir():
        print(f"error: input folder not found: {args.input}", file=sys.stderr)
        return 2

    files = collect_images(args.input, False)
    if not files:
        exts = "/".join(sorted(e.lstrip(".") for e in SUPPORTED_EXT))
        print(f"error: no {exts} images found in {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output
    if out_dir.resolve() == args.input.resolve():
        print("error: output folder must differ from the input folder", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    lab_cfg: dict[str, Any] = dict(LAB_DEFAULTS)
    for key in ("chroma_gain", "lc_gain", "grain", "halation", "dehaze_strength"):
        value = getattr(args, key)
        if value is not None:
            lab_cfg[key] = value

    print(f"experiment.py -- {len(files)} image(s) from {args.input} -> {out_dir}")
    print(f"variants per frame: {', '.join(VARIANTS)}")

    stems = resolve_output_stems(files)
    results: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(process_experiment, f, out_dir, DEFAULTS, lab_cfg, stems[f]): f
                for f in files}
        for fut in concurrent.futures.as_completed(futs):
            f = futs[fut]
            try:
                results[str(f)] = fut.result()
            except Exception as e:
                failures.append((f.name, f"{type(e).__name__}: {e}"))
                traceback.print_exc()

    ordered = [results[str(f)] for f in files if str(f) in results]
    for r in ordered:
        s = r["stats"]
        print(f"  {r['stem']:<12} class={s.preset:<8} chroma_gain={r['params']['chroma_gain']} "
              f"lc_gain={r['params']['lc_gain']}")

    if not args.no_contact_sheet and ordered:
        width = DEFAULTS["contact_thumb_width"]
        rows = []
        for r in ordered:
            row = {
                "name": r["stem"],
                "preset": r["stats"].preset,
                "cast": r["stats"].cast_severity,
                "cast_ratio": r["stats"].cast_ratio,
                "min_range": r["stats"].min_channel_range,
                "recoverability": r["stats"].recoverability,
                "original": thumb(r["original"], width),
            }
            for key in SHEET_COLUMNS[1:]:
                arr = np.clip(r["outputs"][key] * 255.0, 0, 255).astype(np.uint8)
                row[key] = thumb(arr, width)
            rows.append(row)
        sheet = out_dir / "contact_sheet.jpg"
        build_contact_sheet(rows, sheet, width, columns=SHEET_COLUMNS, titles=SHEET_TITLES,
                            heading="35mm film -- experimental looks")
        print(f"\ncontact sheet -> {sheet}")

    print(f"\ndone: {len(ordered)} frames, {len(ordered) * len(VARIANTS)} files -> {out_dir}")
    for name, err in failures:
        print(f"  FAILED {name}: {err}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
