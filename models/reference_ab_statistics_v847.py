"""Geometry-independent robust AB statistics for the V2.47 appearance test."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab


def _median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        rows.append(pixels.median() if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def reference_ab_statistics(reference_rgb: torch.Tensor, reference_hair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    lab = rgb_to_lab(reference_rgb)
    ab = lab[:, 1:]
    chroma = ab.norm(dim=1, keepdim=True)
    median_ab = torch.stack((_median(ab[:, :1], reference_hair_mask), _median(ab[:, 1:2], reference_hair_mask)), dim=1)
    median_chroma = _median(chroma, reference_hair_mask)
    hue = torch.atan2(ab[:, 1:2], ab[:, 0:1])
    median_hue = _median(hue, reference_hair_mask)
    return {"reference_median_ab": median_ab, "reference_chroma_median": median_chroma, "reference_stable_hue": median_hue, "reference_stable_hue_unit": torch.stack((median_hue.cos(), median_hue.sin()), dim=1), "reference_lab": lab}


__all__ = ["reference_ab_statistics"]
