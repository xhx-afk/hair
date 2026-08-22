"""V2.46 carrier-safe, band-limited illumination field.

The module owns only the hair appearance residual.  Ownership is softened
before constructing targets and every L residual is low-pass filtered again
before it is added to the frozen carrier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.v245_death_test_common import dilate


def _blur(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    return F.avg_pool2d(value.float(), 2 * radius + 1, stride=1, padding=radius)


def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        rows.append(pixels.median() if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


class HairLowFrequencyIlluminationFieldV846:
    def __init__(self, *, low_radius: int = 11, ownership_radius: int = 5,
                 residual_radius: int = 5, max_delta_l: float = 8.0,
                 anchor_residual_gain: float = 0.8) -> None:
        self.low_radius = int(low_radius)
        self.ownership_radius = int(ownership_radius)
        self.residual_radius = int(residual_radius)
        self.max_delta_l = float(max_delta_l)
        self.anchor_residual_gain = float(anchor_residual_gain)

    def __call__(self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 carrier_rgb: torch.Tensor, target_hair_mask: torch.Tensor,
                 source_hair_mask: torch.Tensor, reference_l_center: torch.Tensor,
                 disable_l: bool = False) -> dict[str, torch.Tensor]:
        base_l = rgb_to_lab(base_rgb)[:, :1]
        anchor_l = rgb_to_lab(strong_anchor_rgb)[:, :1]
        carrier_lab = rgb_to_lab(carrier_rgb)
        carrier_l, carrier_ab = carrier_lab[:, :1], carrier_lab[:, 1:]
        carrier_l_low = _blur(carrier_l, self.low_radius)

        target_soft = _blur(target_hair_mask.float().clamp(0, 1), self.ownership_radius).clamp(0, 1)
        source_soft = _blur(source_hair_mask.float().clamp(0, 1), self.ownership_radius).clamp(0, 1)
        existing = target_soft * source_soft
        new = target_soft * (1.0 - source_soft)
        weight_sum = (existing + new).clamp_min(1e-6)
        existing = existing / weight_sum
        new = new / weight_sum

        base_l_low = _blur(base_l, self.low_radius)
        existing_target = _blur(base_l_low * target_soft, self.residual_radius)
        existing_target = existing_target / _blur(target_soft, self.residual_radius).clamp_min(1e-4)
        anchor_l_low = _blur(anchor_l, self.low_radius)
        anchor_center = _masked_median(anchor_l_low, target_hair_mask).view(-1, 1, 1, 1)
        reference_center = reference_l_center.float()
        if reference_center.dim() == 1:
            reference_center = reference_center.view(-1, 1, 1, 1)
        elif reference_center.dim() == 2:
            reference_center = reference_center.view(reference_center.size(0), reference_center.size(1), 1, 1)
        if reference_center.shape[-2:] != carrier_l.shape[-2:]:
            reference_center = reference_center.expand(-1, -1, carrier_l.size(2), carrier_l.size(3))
        anchor_residual = anchor_l_low - anchor_center
        new_target = reference_center + self.anchor_residual_gain * anchor_residual
        desired_hair_l_low = existing * existing_target + new * new_target
        desired_hair_l_low = _blur(desired_hair_l_low, self.residual_radius)
        delta_l_raw = (desired_hair_l_low - carrier_l_low) * target_soft
        delta_l_bandlimited = _blur(delta_l_raw, self.residual_radius).clamp(-self.max_delta_l, self.max_delta_l)
        if disable_l:
            delta_l_bandlimited = torch.zeros_like(delta_l_bandlimited)
        final_l = carrier_l + delta_l_bandlimited

        return {
            "carrier_l": carrier_l, "carrier_l_low": carrier_l_low,
            "existing_target_l_low": existing_target, "new_target_l_low": new_target,
            "ownership_soft": target_soft, "existing_weight": existing,
            "new_weight": new, "desired_l_low": desired_hair_l_low,
            "delta_l_low_raw": delta_l_raw, "delta_l_low_bandlimited": delta_l_bandlimited,
            "final_l": final_l, "target_hair_soft": target_soft,
        }


__all__ = ["HairLowFrequencyIlluminationFieldV846"]
