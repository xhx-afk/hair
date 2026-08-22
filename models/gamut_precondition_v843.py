"""Precondition Lab chroma before the final gamut binary search."""

from __future__ import annotations

import torch

from models.gamut_safe_lab_v841 import lab_to_rgb_unclamped_v841


def gamut_precondition_v843(lab: torch.Tensor, *, iterations: int = 6,
                            safety_margin: float = 0.985) -> tuple[torch.Tensor, torch.Tensor]:
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
    l, ab = lab[:, :1], lab[:, 1:]
    raw = lab_to_rgb_unclamped_v841(lab)
    unsafe = ((raw < 0.0) | (raw > 1.0)).any(dim=1, keepdim=True)
    low, high = torch.zeros_like(l), torch.ones_like(l)
    for _ in range(max(int(iterations), 1)):
        mid = (low + high) * 0.5
        candidate = lab_to_rgb_unclamped_v841(torch.cat((l, ab * mid), dim=1))
        valid = ((candidate >= 0.0) & (candidate <= 1.0)).all(dim=1, keepdim=True)
        low = torch.where(valid, mid, low)
        high = torch.where(valid, high, mid)
    scale = torch.where(unsafe, (low * float(safety_margin)).clamp(0, 1), torch.ones_like(low))
    return torch.cat((l, ab * scale), dim=1), scale


__all__ = ["gamut_precondition_v843"]
