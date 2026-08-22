"""V2.41 target field: luminance changes chroma magnitude only."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class TargetStableHueChromaFieldV841:
    def __init__(self, *, low_radius: int = 11,
                 achromatic_hue_strength: float = 0.05) -> None:
        self.low_radius = int(low_radius)
        self.achromatic_hue_strength = float(achromatic_hue_strength)

    @staticmethod
    def _quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        values = []
        for batch in range(value.size(0)):
            pixels = value[batch, 0][mask[batch, 0] > 0.5]
            values.append(torch.quantile(pixels, q) if pixels.numel() else value.new_tensor(50.0))
        return torch.stack(values).view(-1, 1, 1, 1)

    def __call__(self, *, target_illumination_rgb: torch.Tensor,
                 target_hair_mask: torch.Tensor, palette: dict[str, torch.Tensor],
                 target_hair_alpha: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        target_l = gaussian_blur_v829(rgb_to_lab(target_illumination_rgb)[:, :1], self.low_radius)
        mask = target_hair_mask.float().clamp(0, 1)
        lo = self._quantile(target_l, mask, 0.05)
        hi = self._quantile(target_l, mask, 0.95)
        relative_l = ((target_l - lo) / (hi - lo).clamp_min(1.0)).clamp(0, 1)
        t_mid = ((relative_l - 0.35) / 0.35).clamp(0, 1)
        t_high = ((relative_l - 0.70) / 0.30).clamp(0, 1)
        w_shadow, w_mid, w_high = 1.0 - t_mid, t_mid - t_high, t_high
        chroma = (w_shadow * palette["shadow_chroma"] +
                  w_mid * palette["mid_chroma"] +
                  w_high * palette["highlight_chroma"])
        unit = palette["stable_hue_unit_ab"]
        unit_field = unit.expand(-1, -1, target_l.size(-2), target_l.size(-1))
        target_ab = unit_field * chroma
        strength_x = ((palette["global_chroma"] - 1.5) / (6.0 - 1.5)).clamp(0, 1)
        smooth = strength_x * strength_x * (3.0 - 2.0 * strength_x)
        hue_strength = self.achromatic_hue_strength + (1.0 - self.achromatic_hue_strength) * smooth
        hue_strength = hue_strength * (0.5 + 0.5 * palette["valid_hue_fraction"].clamp(0, 1))
        target_ab = target_ab * hue_strength
        alpha = mask if target_hair_alpha is None else target_hair_alpha.float().clamp(0, 1)
        return {
            "target_l_low": target_l,
            "target_relative_l": relative_l,
            "target_relative_weights": torch.cat((w_shadow, w_mid, w_high), dim=1),
            "reference_chroma_low": chroma,
            "target_chroma_field": chroma,
            "target_ab_low": target_ab,
            "target_ab": target_ab,
            "stable_hue_unit_ab_field": unit_field,
            "achromatic_hue_strength_map": hue_strength,
            "chroma_confidence": alpha * palette["palette_reliability"].clamp(0, 1),
            "target_hair_alpha": alpha,
        }


__all__ = ["TargetStableHueChromaFieldV841"]
