"""V2.43 low-frequency illumination ownership for existing versus new hair."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab


def _blur(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    return F.avg_pool2d(value, 2 * radius + 1, stride=1, padding=radius)


def new_hair_illumination_v843(*, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                               target_hair_mask: torch.Tensor, source_hair_mask: torch.Tensor | None,
                               reference_l_center: torch.Tensor, radius: int = 11) -> dict[str, torch.Tensor]:
    target = target_hair_mask.float().clamp(0, 1)
    source = torch.zeros_like(target) if source_hair_mask is None else source_hair_mask.float().clamp(0, 1)
    existing = target * source
    new = target * (1.0 - source)
    base_l = _blur(rgb_to_lab(base_rgb)[:, :1], radius)
    anchor_l = _blur(rgb_to_lab(strong_anchor_rgb)[:, :1], radius)
    values = []
    for batch_index in range(anchor_l.size(0)):
        pixels = anchor_l[batch_index, 0][target[batch_index, 0] > 0.5]
        values.append(pixels.median() if pixels.numel() else anchor_l[batch_index, 0].median())
    anchor_center = torch.stack(values).view(-1, 1, 1, 1)
    anchor_residual = anchor_l - anchor_center
    new_l = reference_l_center.float() + 0.8 * anchor_residual
    mapped = existing * base_l + new * new_l + (1.0 - target) * base_l
    return {
        "existing_hair": existing,
        "new_hair": new,
        "anchor_l_low": anchor_l,
        "anchor_center": anchor_center,
        "anchor_l_residual": anchor_residual,
        "existing_hair_l_low": existing * base_l,
        "new_hair_l_low": new * new_l,
        "mapped_l_low": mapped,
    }


__all__ = ["new_hair_illumination_v843"]
