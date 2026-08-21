"""Unclamped Lab to RGB conversion and local chroma safety."""

from __future__ import annotations

import torch


def _linear_to_srgb_unclamped(image: torch.Tensor) -> torch.Tensor:
    nonlinear = 1.055 * image.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055
    return torch.where(image <= 0.0031308, 12.92 * image, nonlinear)


def lab_to_rgb_unclamped_v840(lab: torch.Tensor) -> torch.Tensor:
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
    l, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (l + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    def finv(t: torch.Tensor) -> torch.Tensor:
        t3 = t.pow(3)
        return torch.where(t3 > eps, t3, (116.0 * t - 16.0) / kappa)
    x, y, z = finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return _linear_to_srgb_unclamped(torch.cat((r, g, blue), dim=1))


def gamut_safe_lab_to_rgb_v840(lab: torch.Tensor, *, iterations: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = lab_to_rgb_unclamped_v840(lab)
    unsafe = ((raw < 0.0) | (raw > 1.0)).any(dim=1, keepdim=True)
    low, high = torch.zeros_like(lab[:, :1]), torch.ones_like(lab[:, :1])
    scale = high.clone()
    for _ in range(max(int(iterations), 1)):
        mid = (low + high) * 0.5
        candidate = lab_to_rgb_unclamped_v840(torch.cat((lab[:, :1], lab[:, 1:] * mid), dim=1))
        valid = ((candidate >= 0.0) & (candidate <= 1.0)).all(dim=1, keepdim=True)
        high = torch.where(valid, mid, high)
        low = torch.where(valid, low, mid)
        scale = torch.where(valid, mid, scale)
    safe = torch.where(unsafe, lab_to_rgb_unclamped_v840(torch.cat((lab[:, :1], lab[:, 1:] * scale), dim=1)), raw)
    clip_ratio = unsafe.float()
    return safe.clamp(0, 1), scale, clip_ratio


__all__ = ["lab_to_rgb_unclamped_v840", "gamut_safe_lab_to_rgb_v840"]
