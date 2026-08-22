"""Independent flyaway candidate extraction for the V2.45.1 matte test.

This module intentionally consumes only the frozen strong-anchor RGB and the
coarse target hair mask.  It must remain independent from the matte probe's
alpha, score, and threshold so the recall metric is not self-referential.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v245_death_test_common import dilate


def _sobel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = value.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = value.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    return F.conv2d(value, kx, padding=1), F.conv2d(value, ky, padding=1)


def independent_flyaway_candidate(
    strong_anchor_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor,
    source_skin_mask: torch.Tensor | None = None,
    *,
    radius: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(candidate, score)`` without using probe-produced tensors."""
    hair = target_hair_mask.float().clamp(0, 1)
    anchor = strong_anchor_rgb.float().clamp(0, 1)
    anchor_l = (0.2126 * anchor[:, 0:1] + 0.7152 * anchor[:, 1:2] + 0.0722 * anchor[:, 2:3])
    gx, gy = _sobel(anchor_l)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-8)
    # Structure tensor anisotropy: coherent strand-like gradients approach 1.
    jxx = F.avg_pool2d(gx.square(), 5, 1, 2)
    jyy = F.avg_pool2d(gy.square(), 5, 1, 2)
    jxy = F.avg_pool2d(gx * gy, 5, 1, 2)
    delta = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1e-8)
    anisotropy = (delta / (jxx + jyy + 1e-6)).clamp(0, 1)
    local = (anchor_l - F.avg_pool2d(anchor_l, 5, 1, 2)).abs()
    local_scale = local.flatten(1).amax(1).view(-1, 1, 1, 1).clamp_min(1e-5)
    normalized_contrast = (local / local_scale).clamp(0, 1)
    grad_scale = grad.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
    normalized_grad = (grad / grad_scale).clamp(0, 1)
    score = 0.45 * anisotropy + 0.35 * normalized_contrast + 0.20 * normalized_grad
    support_ring = dilate(hair, radius) * (1.0 - hair)
    threshold = torch.full_like(score, 0.65)
    if source_skin_mask is not None:
        threshold = torch.where(source_skin_mask.float().clamp(0, 1) > 0.5, torch.full_like(score, 0.78), threshold)
    candidate = support_ring * (score > threshold).float()
    return candidate, score


__all__ = ["independent_flyaway_candidate"]
