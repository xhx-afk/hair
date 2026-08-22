"""Rebuild target hair chroma from target-relative luminance, not XY transfer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class TargetRelativeChromaFieldV837:
    def __init__(self, *, low_radius: int = 11, low_quantiles: tuple[float, float] = (0.05, 0.95)) -> None:
        if low_radius < 1 or not 0 <= low_quantiles[0] < low_quantiles[1] <= 1:
            raise ValueError("V2.37 target-relative field parameters are invalid")
        self.low_radius = int(low_radius)
        self.low_quantiles = low_quantiles

    @staticmethod
    def _quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        rows = []
        for batch in range(value.size(0)):
            pixels = value[batch, 0][mask[batch, 0] > 0.5]
            rows.append(torch.quantile(pixels, q) if pixels.numel() else value.new_tensor(0.0))
        return torch.stack(rows).view(-1, 1, 1, 1)

    def __call__(
        self, *, target_illumination_rgb: torch.Tensor, target_hair_mask: torch.Tensor,
        palette: dict[str, torch.Tensor], target_hair_alpha: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_lab = rgb_to_lab(target_illumination_rgb)
        target_l_low = gaussian_blur_v829(target_lab[:, :1], self.low_radius)
        mask = target_hair_mask.float().clamp(0, 1)
        lo = self._quantile(target_l_low, mask, self.low_quantiles[0])
        hi = self._quantile(target_l_low, mask, self.low_quantiles[1])
        relative_l = ((target_l_low - lo) / (hi - lo).clamp_min(1.0)).clamp(0, 1)
        shadow = palette["shadow_ab"]
        mid = palette["mid_ab"]
        highlight = palette["highlight_ab"]
        # Smooth piecewise interpolation: shadow->mid around .35, mid->highlight around .70.
        t_mid = ((relative_l - 0.35) / 0.35).clamp(0, 1)
        t_high = ((relative_l - 0.70) / 0.30).clamp(0, 1)
        w_shadow = 1.0 - t_mid
        w_mid = t_mid - t_high
        w_highlight = t_high
        weights = torch.cat((w_shadow, w_mid, w_highlight), dim=1)
        target_ab = w_shadow * shadow + w_mid * mid + w_highlight * highlight
        alpha = mask if target_hair_alpha is None else target_hair_alpha.float().clamp(0, 1)
        chroma_confidence = alpha * palette["palette_reliability"].clamp(0, 1)
        return {
            "target_l_low": target_l_low,
            "target_relative_l": relative_l,
            "target_relative_weights": weights,
            "target_ab": target_ab,
            "chroma_confidence": chroma_confidence,
            "target_hair_alpha": alpha,
        }


__all__ = ["TargetRelativeChromaFieldV837"]
