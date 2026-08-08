from __future__ import annotations

from pathlib import Path

from .analysis import FrameStats

PCT_KEYS = ("p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99")


def format_report(stats: FrameStats, params: dict[str, dict]) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"\n{'=' * 78}")
    add(f"  {Path(stats.path).name}   [{stats.width}x{stats.height}]")
    add(f"{'=' * 78}")

    add("  RGB percentiles BEFORE correction (8-bit):")
    header = f"{'ch':<3}" + "".join(f"{k:>7}" for k in PCT_KEYS) + f"{'p99.9':>8}"
    add(f"      {header}")
    for c in "RGB":
        p = stats.percentiles[c]
        add(f"      {c:<3}" + "".join(f"{p[k]:>7.0f}" for k in PCT_KEYS) + f"{p['p99.9']:>8.0f}")

    add("  Usable range per channel (white point - black point, 8-bit codes):")
    add("      " + "   ".join(f"{c}={stats.channel_range[c]:6.1f}" for c in "RGB")
        + f"    -> worst = {stats.min_channel_range:.0f} (~{stats.effective_bits:.1f} bits)")

    add("  Correction multipliers (linear light):")
    lv = params.get("neutral", {})
    if "levels_gain_linear" in lv:
        add("      gain     R={:.3f}  G={:.3f}  B={:.3f}".format(*lv["levels_gain_linear"]))
        add("      black    R={:.4f}  G={:.4f}  B={:.4f}".format(*lv["levels_black_linear"]))
        add("      midtone  R={:.3f}  G={:.3f}  B={:.3f}   (per-channel gamma, 3rd match point)"
            .format(*lv["midtone_gamma"]))
        add(f"      neutrality={lv.get('neutrality')}  LAB residual cast a*/b* "
            f"{lv.get('lab_cast_bias_ab')} -> removed {lv.get('lab_cast_applied_ab')}")
        add(f"      tone: black={lv.get('tone_black_point')} white={lv.get('tone_white_point')} "
            f"S={lv.get('tone_scurve')} gamma={lv.get('tone_gamma')}")
        add(f"      CLAHE clip={lv.get('clahe_clip')} blend={lv.get('clahe_blend')}  "
            f"sat={lv.get('saturation')}  chromaNR={lv.get('chroma_denoise')}  "
            f"sharpen={lv.get('sharpen_amount')}")
        add(f"      chroma confidence gate: mean scale={lv.get('chroma_confidence_mean')} "
            f"(1.0 = colour trusted everywhere)")
        add(f"      fog flatten: {lv.get('fog_flatten')}")

    bw = params.get("bw", {})
    if "bw_channel_weights" in bw:
        add("      B&W channel mix  R={:.3f}  G={:.3f}  B={:.3f}  (weighted by information)"
            .format(*bw["bw_channel_weights"]))

    add(f"  Colour cast: {stats.cast_severity.upper()}  ({stats.cast_hue}), "
        f"channel-gain ratio {stats.cast_ratio:.2f}x")
    add(f"  Clipping:  shadows={'YES' if stats.shadows_clipped else 'no'}  "
        f"highlights={'YES' if stats.highlights_clipped else 'no'}")
    add("      at 0:   " + "  ".join(f"{c}={stats.clipped_low[c]:5.2f}%" for c in "RGB")
        + "     at 255: " + "  ".join(f"{c}={stats.clipped_high[c]:5.2f}%" for c in "RGB"))
    add(f"  Underexposed / thin: {'YES' if stats.underexposed else 'no'}    "
        f"detail score={stats.detail_score:.2f}    fog unevenness={stats.fog_unevenness:.1f}")
    add(f"  Correction class: {stats.preset.upper()}      "
        f"Recoverability: {stats.recoverability.upper()}")
    for n in stats.notes:
        add(f"    ! {n}")

    return "\n".join(lines)
