from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn, idctn

from ..color_space import linear_to_srgb, rgb_to_lab, srgb_to_linear

RECON_DEFAULTS = {
    "seam_z": 12.0,
    "seam_max": 4,
    "seam_halfwidth": 3,
    "seam_blur": 2.0,
    "spot_max_area": 40,
    "spot_z": 5.0,
    "spot_radius": 3,
}


def poisson_solve(div: np.ndarray) -> np.ndarray:
    m, n = div.shape
    coeff = dctn(div, type=2, norm="ortho")
    i = np.arange(m).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    denom = 2.0 * np.cos(np.pi * i / m) + 2.0 * np.cos(np.pi * j / n) - 4.0
    denom[0, 0] = 1.0
    out = coeff / denom
    out[0, 0] = 0.0
    return idctn(out, type=2, norm="ortho")


def forward_gradients(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(plane)
    gy = np.zeros_like(plane)
    gx[:, :-1] = plane[:, 1:] - plane[:, :-1]
    gy[:-1, :] = plane[1:, :] - plane[:-1, :]
    return gx, gy


def divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    d = np.zeros_like(gx)
    d[:, 1:] += gx[:, 1:] - gx[:, :-1]
    d[:, 0] += gx[:, 0]
    d[1:, :] += gy[1:, :] - gy[:-1, :]
    d[0, :] += gy[0, :]
    return d


def _profile_peaks(profile: np.ndarray, z_thresh: float, limit: int) -> list[int]:
    med = float(np.median(profile))
    mad = float(np.median(np.abs(profile - med))) + 1e-6
    z = (profile - med) / mad
    idx = np.where(z > z_thresh)[0]
    if idx.size == 0:
        return []
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 4:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    scored = sorted(((float(z[g].max()), int(np.mean(g))) for g in groups), reverse=True)
    edge = max(6, int(0.01 * profile.size))
    kept = [pos for _, pos in scored if edge < pos < profile.size - edge]
    return kept[:limit]


def detect_seams(rgb: np.ndarray, cfg: dict | None = None) -> tuple[list[int], list[int]]:
    cfg = {**RECON_DEFAULTS, **(cfg or {})}
    lightness = cv2.GaussianBlur(rgb_to_lab(rgb)[..., 0], (0, 0), float(cfg["seam_blur"]))
    col_profile = np.abs(np.diff(lightness, axis=1)).mean(axis=0)
    row_profile = np.abs(np.diff(lightness, axis=0)).mean(axis=1)
    limit = int(cfg["seam_max"])
    cols = _profile_peaks(col_profile, float(cfg["seam_z"]), limit)
    rows = _profile_peaks(row_profile, float(cfg["seam_z"]), limit)
    return cols, rows


def remove_seams(rgb: np.ndarray, cols: list[int], rows: list[int],
                 cfg: dict | None = None) -> np.ndarray:
    if not cols and not rows:
        return rgb
    cfg = {**RECON_DEFAULTS, **(cfg or {})}
    half = int(cfg["seam_halfwidth"])
    lin = srgb_to_linear(rgb)
    out = np.empty_like(lin)
    h, w = lin.shape[:2]

    for c in range(3):
        plane = lin[..., c]
        gx, gy = forward_gradients(plane)
        for col in cols:
            gx[:, max(0, col - half):min(w, col + half + 1)] = 0.0
        for row in rows:
            gy[max(0, row - half):min(h, row + half + 1), :] = 0.0
        solved = poisson_solve(divergence(gx, gy))
        out[..., c] = solved + (float(plane.mean()) - float(solved.mean()))

    return linear_to_srgb(np.clip(out, 0.0, 1.0))


def detect_spots(rgb: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    cfg = {**RECON_DEFAULTS, **(cfg or {})}
    lightness = rgb_to_lab(rgb)[..., 0]
    median = cv2.medianBlur(lightness.astype(np.float32), 5)
    residual = lightness - median
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
    candidate = (np.abs(residual) > float(cfg["spot_z"]) * sigma).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    mask = np.zeros_like(candidate)
    max_area = int(cfg["spot_max_area"])
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            mask[labels == i] = 255
    return mask


def spot_defects(rgb: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, int]:
    cfg = {**RECON_DEFAULTS, **(cfg or {})}
    mask = detect_spots(rgb, cfg)
    count = int(cv2.connectedComponentsWithStats(mask, 8)[0]) - 1
    if count <= 0:
        return rgb, 0
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    fixed = cv2.inpaint(rgb8, mask, int(cfg["spot_radius"]), cv2.INPAINT_TELEA)
    return (fixed.astype(np.float32) / 255.0), count
