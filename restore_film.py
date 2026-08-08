#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from m35 import (DEFAULTS, PRESETS, SUPPORTED_EXT, build_contact_sheet, collect_images,
                 process_file, resolve_output_stems)

TUNING_FLAGS = [
    ("neutrality", float, "0 = keep the cast, 1 = full per-channel match"),
    ("saturation", float, None),
    ("scurve-amount", float, None),
    ("clahe-clip", float, None),
    ("clahe-blend", float, None),
    ("sharpen-amount", float, None),
    ("chroma-denoise", float, None),
    ("luma-denoise", float, None),
    ("black-percentile", float, None),
    ("white-percentile", float, None),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="restore_film.py",
        description="Non-generative restoration of badly scanned 35mm colour-negative film.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="folder of scans (read-only, never modified)")
    ap.add_argument("output", type=Path, nargs="?", default=Path("./output"),
                    help="folder for restored images")
    ap.add_argument("--recursive", action="store_true", help="descend into sub-folders")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="parallel worker threads")
    ap.add_argument("--quality", type=int, default=None, help="output JPEG quality (1-100)")
    ap.add_argument("--no-contact-sheet", action="store_true")
    ap.add_argument("--report", type=Path, default=None,
                    help="also write the full diagnostics as JSON to this path")

    t = ap.add_argument_group("tuning (override the auto-selected preset)")
    t.add_argument("--force-preset", choices=sorted(PRESETS), default=None,
                   help="apply one correction class to every frame")
    t.add_argument("--fog-flatten", choices=["auto", "on", "off"], default=None)
    t.add_argument("--predenoise", choices=["auto", "on", "off"], default=None)
    for name, kind, help_text in TUNING_FLAGS:
        t.add_argument(f"--{name}", type=kind, default=None, help=help_text)
    return ap.parse_args(argv)


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "force_preset": args.force_preset,
        "fog_flatten": args.fog_flatten,
        "predenoise": args.predenoise,
    }
    for name, _kind, _help in TUNING_FLAGS:
        key = name.replace("-", "_")
        overrides[key] = getattr(args, key)
    return overrides


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.input.is_dir():
        print(f"error: input folder not found: {args.input}", file=sys.stderr)
        return 2

    files = collect_images(args.input, args.recursive)
    if not files:
        exts = "/".join(sorted(e.lstrip(".") for e in SUPPORTED_EXT))
        print(f"error: no {exts} images found in {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output
    if out_dir.resolve() == args.input.resolve():
        print("error: output folder must differ from the input folder "
              "(originals are never modified)", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = dict(DEFAULTS)
    if args.quality is not None:
        base_cfg["jpeg_quality"] = args.quality

    overrides = collect_overrides(args)
    want_sheet = not args.no_contact_sheet

    print(f"restore_film.py -- {len(files)} image(s) from {args.input} -> {out_dir}")
    print("(originals are opened read-only and never written to)")

    results: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []
    stems = resolve_output_stems(files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(process_file, f, out_dir, base_cfg, overrides,
                          want_sheet, stems[f]): f for f in files}
        for fut in concurrent.futures.as_completed(futs):
            f = futs[fut]
            try:
                results[str(f)] = fut.result()
            except Exception as e:
                failures.append((f.name, f"{type(e).__name__}: {e}"))
                traceback.print_exc()

    ordered = [results[str(f)] for f in files if str(f) in results]
    for r in ordered:
        print(r["report"])

    if want_sheet and ordered:
        rows = [r["sheet_row"] for r in ordered if r["sheet_row"]]
        sheet_path = out_dir / "contact_sheet.jpg"
        build_contact_sheet(rows, sheet_path, base_cfg["contact_thumb_width"])
        print(f"\ncontact sheet -> {sheet_path}")

    if args.report:
        payload = [{"stats": r["stats"].to_dict(), "params": r["params"]} for r in ordered]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2))
        print(f"json diagnostics -> {args.report}")

    print(f"\ndone: {len(ordered)} restored, {len(failures)} failed. "
          f"{len(ordered) * 3} files written to {out_dir}")
    for name, err in failures:
        print(f"  FAILED {name}: {err}", file=sys.stderr)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
