"""Robust piecewise reference tone mapping for target low-frequency L."""

from __future__ import annotations

import torch


class ReferenceToneMapperV840:
    def __init__(self, *, scale_min: float = 0.60, scale_max: float = 1.40) -> None:
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    @staticmethod
    def _quantiles(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = []
        for batch in range(value.size(0)):
            pixels = value[batch, 0][mask[batch, 0] > 0.5]
            if pixels.numel() == 0:
                pixels = value.new_tensor([50.0])
            output.append(torch.quantile(pixels, torch.tensor([0.25, 0.50, 0.75], device=pixels.device, dtype=pixels.dtype)))
        return torch.stack(output).view(-1, 3, 1, 1)

    def __call__(self, *, target_l_low: torch.Tensor, hair_mask: torch.Tensor,
                 reference_tone: dict[str, torch.Tensor], return_aux: bool = False):
        target_q = self._quantiles(target_l_low, hair_mask)
        ref_q = torch.cat((reference_tone["l_q25"], reference_tone["l_q50"], reference_tone["l_q75"]), dim=1)
        eps = target_l_low.new_tensor(1e-4)
        scale_neg = ((ref_q[:, 1:2] - ref_q[:, 0:1]) / (target_q[:, 1:2] - target_q[:, 0:1] + eps)).clamp(self.scale_min, self.scale_max)
        scale_pos = ((ref_q[:, 2:3] - ref_q[:, 1:2]) / (target_q[:, 2:3] - target_q[:, 1:2] + eps)).clamp(self.scale_min, self.scale_max)
        delta = target_l_low - target_q[:, 1:2]
        mapped = torch.where(delta < 0, ref_q[:, 1:2] + delta * scale_neg, ref_q[:, 1:2] + delta * scale_pos)
        if not return_aux:
            return mapped
        return mapped, {
            "target_l_quantiles": target_q,
            "reference_l_quantiles": ref_q,
            "tone_scale_negative": scale_neg,
            "tone_scale_positive": scale_pos,
            "mapped_l_low": mapped,
        }


__all__ = ["ReferenceToneMapperV840"]
