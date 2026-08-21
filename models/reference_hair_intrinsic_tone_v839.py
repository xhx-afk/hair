"""Geometry-independent intrinsic LAB tone statistics for reference hair."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.reference_hair_chroma_palette_v837 import ReferenceHairChromaPaletteV837


class ReferenceHairTonePaletteV839(ReferenceHairChromaPaletteV837):
    """Keep the V2.37 AB palette and add robust, intrinsic L prototypes."""

    def __call__(self, *, reference_rgb: torch.Tensor, reference_hair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        result = super().__call__(reference_rgb=reference_rgb, reference_hair_mask=reference_hair_mask)
        lab = rgb_to_lab(reference_rgb)
        mask = reference_hair_mask.float().clamp(0, 1) > 0.5
        centers, scales, quantiles = [], [], []
        l_prototypes = [[], [], []]
        for batch in range(lab.size(0)):
            pixels = lab[batch].permute(1, 2, 0)[mask[batch, 0]]
            if pixels.numel() == 0:
                pixels = lab.new_zeros((1, 3))
            center = pixels.median(dim=0).values
            deviation = (pixels[:, 1:] - center[1:]).norm(dim=1)
            mad = deviation.median().clamp_min(1.0)
            keep = deviation <= self.mad_scale * mad
            filtered = pixels[keep] if keep.any() else pixels
            l = filtered[:, 0]
            q = torch.quantile(l, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=l.device, dtype=l.dtype))
            quantiles.append(q)
            centers.append(q[2])
            scales.append((q[3] - q[1]).clamp_min(1.0))
            raw_q = torch.quantile(l, torch.tensor([0.25, 0.75], device=l.device, dtype=l.dtype))
            groups = (l <= raw_q[0]), ((l > raw_q[0]) & (l < raw_q[1])), (l >= raw_q[1])
            for index, group in enumerate(groups):
                values = l[group]
                l_prototypes[index].append(values.median() if values.numel() else q[2])
        q = torch.stack(quantiles).view(-1, 5, 1, 1)
        result.update({
            "shadow_l": torch.stack(l_prototypes[0]).view(-1, 1, 1, 1),
            "mid_l": torch.stack(l_prototypes[1]).view(-1, 1, 1, 1),
            "highlight_l": torch.stack(l_prototypes[2]).view(-1, 1, 1, 1),
            "l_q10": q[:, 0:1], "l_q25": q[:, 1:2], "l_q50": q[:, 2:3],
            "l_q75": q[:, 3:4], "l_q90": q[:, 4:5],
            "intrinsic_l_center": torch.stack(centers).view(-1, 1, 1, 1),
            "intrinsic_l_scale": torch.stack(scales).view(-1, 1, 1, 1),
        })
        result["intrinsic_tone_preview_rgb"] = lab_to_rgb(torch.cat((result["intrinsic_l_center"], result["global_ab"]), dim=1)).clamp(0, 1)
        return result


__all__ = ["ReferenceHairTonePaletteV839"]
