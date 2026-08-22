"""V2.41.1 target illumination ownership."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class TargetIlluminationExtractorV8411:
    def __init__(self, *, low_radius: int = 11) -> None:
        if low_radius < 1:
            raise ValueError("low_radius must be positive")
        self.low_radius = int(low_radius)

    @staticmethod
    def _quantile(value: torch.Tensor, mask: torch.Tensor, q: float, fallback: float) -> torch.Tensor:
        output = []
        for batch in range(value.size(0)):
            pixels = value[batch, 0][mask[batch, 0] > 0.5]
            output.append(torch.quantile(pixels, q) if pixels.numel() else value.new_tensor(fallback))
        return torch.stack(output).view(-1, 1, 1, 1)

    def __call__(self, *, target_illumination_rgb: torch.Tensor,
                 target_hair_mask: torch.Tensor, return_aux: bool = False):
        target_l_raw = rgb_to_lab(target_illumination_rgb)[:, :1]
        target_l_low = gaussian_blur_v829(target_l_raw, self.low_radius)
        mask = target_hair_mask.float().clamp(0, 1)
        center = self._quantile(target_l_low, mask, 0.50, 50.0)
        lo = self._quantile(target_l_low, mask, 0.05, 0.0)
        hi = self._quantile(target_l_low, mask, 0.95, 100.0)
        relative_l = ((target_l_low - lo) / (hi - lo).clamp_min(1.0)).clamp(0, 1)
        # The tone mapper owns the absolute reference level.  Only the
        # target-local, zero-centered illumination variation is allowed back
        # into mapped L so changing Base/SATD lighting cannot shift the tone.
        residual = target_l_low - center
        aux = {
            "target_l_raw": target_l_raw,
            "target_l_low": target_l_low,
            "target_l_center": center,
            "target_l_residual": residual,
            "target_illumination_residual": residual,
            "target_relative_l": relative_l,
            "target_l_q05": lo,
            "target_l_q95": hi,
        }
        return aux if return_aux else target_l_low


__all__ = ["TargetIlluminationExtractorV8411"]
