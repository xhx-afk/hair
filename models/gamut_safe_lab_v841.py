"""V2.41 unclamped Lab conversion with correct local chroma compression."""

from __future__ import annotations

import torch


def _linear_to_srgb_unclamped(image: torch.Tensor) -> torch.Tensor:
    nonlinear = 1.055 * image.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055
    return torch.where(image <= 0.0031308, 12.92 * image, nonlinear)


def lab_to_rgb_unclamped_v841(lab: torch.Tensor) -> torch.Tensor:
    """Convert Lab to RGB without clipping the result."""
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
    l, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (l + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def finv(value: torch.Tensor) -> torch.Tensor:
        cube = value.pow(3)
        return torch.where(cube > eps, cube, (116.0 * value - 16.0) / kappa)

    x, y, z = finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return _linear_to_srgb_unclamped(torch.cat((r, g, blue), dim=1))


def gamut_safe_lab_to_rgb_v841(
    lab: torch.Tensor, *, iterations: int = 8, l_min: float = 1.0,
    l_max: float = 99.0, return_aux: bool = False,
):
    """Compress only unsafe pixels to the largest in-gamut chroma scale.

    The binary search invariant is ``low=valid`` and ``high=unsafe``.  The
    default return mirrors the v2.40 three-tensor API; ``return_aux`` exposes
    all v2.41 diagnostics.
    """
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
    lab_input_l = lab[:, :1]
    l = lab_input_l.clamp(float(l_min), float(l_max))
    ab = lab[:, 1:]
    unclamped = lab_to_rgb_unclamped_v841(torch.cat((l, ab), dim=1))
    unsafe = ((unclamped < 0.0) | (unclamped > 1.0)).any(dim=1, keepdim=True)

    low = torch.zeros_like(l)
    high = torch.ones_like(l)
    for _ in range(max(int(iterations), 1)):
        mid = (low + high) * 0.5
        candidate = lab_to_rgb_unclamped_v841(torch.cat((l, ab * mid), dim=1))
        valid = ((candidate >= 0.0) & (candidate <= 1.0)).all(dim=1, keepdim=True)
        # Search for the largest valid chroma, not the smallest invalid one.
        low = torch.where(valid, mid, low)
        high = torch.where(valid, high, mid)

    scale = torch.where(unsafe, low, torch.ones_like(low))
    safe_lab = torch.cat((l, ab * scale), dim=1)
    safe_rgb = lab_to_rgb_unclamped_v841(safe_lab).clamp(0.0, 1.0)
    compressed = (unsafe & (scale < 0.999)).float()
    heavy_fraction = (unsafe & (scale < 0.5)).float().flatten(1).mean(dim=1)
    diagnostics = {
        "lab_input_l": lab_input_l,
        "lab_safe_l": l,
        "gamut_safe_l": l,
        "gamut_safe_lab": safe_lab,
        "gamut_scale_map": scale,
        "gamut_unsafe_map": unsafe.float(),
        "gamut_compressed_map": compressed,
        "gamut_heavy_compression_map": (unsafe & (scale < 0.5)).float(),
        "gamut_min_scale": scale.flatten(1).amin(dim=1),
        "gamut_unsafe_fraction": unsafe.float().flatten(1).mean(dim=1),
        "gamut_compressed_fraction": compressed.flatten(1).mean(dim=1),
        "gamut_heavy_compression_fraction": heavy_fraction,
        "median_gamut_scale": scale.flatten(1).median(dim=1).values,
        "p10_gamut_scale": torch.quantile(scale.flatten(1), 0.10, dim=1),
    }
    if return_aux:
        return safe_rgb, diagnostics
    return safe_rgb, scale, compressed


__all__ = ["lab_to_rgb_unclamped_v841", "gamut_safe_lab_to_rgb_v841"]
