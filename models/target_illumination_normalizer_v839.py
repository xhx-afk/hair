"""Separate target hair illumination residual from its absolute tone."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class TargetIlluminationNormalizerV839:
    def __init__(self, *, low_radius: int = 11, illumination_scale: float = 0.9) -> None:
        if low_radius < 1 or not 0.0 <= illumination_scale <= 1.5:
            raise ValueError("V2.39 illumination parameters are invalid")
        self.low_radius = int(low_radius)
        self.illumination_scale = float(illumination_scale)

    @staticmethod
    def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = []
        for batch in range(value.size(0)):
            pixels = value[batch, 0][mask[batch, 0] > 0.5]
            values.append(pixels.median() if pixels.numel() else value.new_tensor(50.0))
        return torch.stack(values).view(-1, 1, 1, 1)

    def __call__(self, *, target_illumination_rgb: torch.Tensor, target_hair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        target_l = rgb_to_lab(target_illumination_rgb)[:, :1]
        target_l_low = gaussian_blur_v829(target_l, self.low_radius)
        center = self._masked_median(target_l_low, target_hair_mask.float().clamp(0, 1))
        residual = target_l_low - center
        return {
            "target_l_raw": target_l,
            "target_l_low": target_l_low,
            "target_l_center": center,
            "target_illumination_residual": residual,
            "illumination_scale": target_l.new_tensor([self.illumination_scale]),
        }


__all__ = ["TargetIlluminationNormalizerV839"]
