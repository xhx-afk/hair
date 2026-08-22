"""Robust, geometry-independent chroma palette extracted from reference hair."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab


class ReferenceHairChromaPaletteV837:
    """Extract shadow/mid/highlight Lab AB prototypes using reference L ranks."""

    def __init__(self, *, mad_scale: float = 3.5, min_support: int = 16) -> None:
        if mad_scale <= 0 or min_support < 1:
            raise ValueError("V2.37 palette parameters must be positive")
        self.mad_scale = float(mad_scale)
        self.min_support = int(min_support)

    @staticmethod
    def _masked_quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        result = []
        for batch in range(value.size(0)):
            samples = value[batch].flatten()[mask[batch].flatten() > 0.5]
            result.append(torch.quantile(samples, q) if samples.numel() else value.new_tensor(0.0))
        return torch.stack(result).view(-1, 1, 1, 1)

    def __call__(
        self, *, reference_rgb: torch.Tensor, reference_hair_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        lab = rgb_to_lab(reference_rgb)
        l = lab[:, :1]
        ab = lab[:, 1:]
        mask = reference_hair_mask.float().clamp(0, 1)
        valid = mask > 0.5
        counts = valid.flatten(1).sum(1, keepdim=True).float()
        global_ab = []
        reliability = []
        prototypes = [[], [], []]
        for batch in range(lab.size(0)):
            pixels = ab[batch].permute(1, 2, 0)[valid[batch, 0]]
            lp = l[batch, 0][valid[batch, 0]]
            if pixels.size(0) == 0:
                pixels = ab.new_zeros((1, 2))
                lp = l.new_zeros((1,))
            center = pixels.median(dim=0).values
            deviation = (pixels - center).norm(dim=1)
            mad = deviation.median().clamp_min(1.0)
            keep = deviation <= self.mad_scale * mad
            filtered = pixels[keep] if keep.any() else pixels
            global_ab.append(filtered.median(dim=0).values)
            outlier_ratio = 1.0 - keep.float().mean()
            support_score = (counts[batch, 0] / float(self.min_support)).clamp(max=1.0)
            reliability.append((support_score * (1.0 - outlier_ratio)).clamp(0, 1))
            q = torch.quantile(lp, torch.tensor([0.25, 0.75], device=lp.device, dtype=lp.dtype))
            groups = (lp <= q[0]), ((lp > q[0]) & (lp < q[1])), (lp >= q[1])
            for index, group in enumerate(groups):
                group_pixels = pixels[group & keep]
                if group_pixels.size(0) == 0:
                    group_pixels = filtered
                prototypes[index].append(group_pixels.median(dim=0).values)
        result = {
            "shadow_ab": torch.stack(prototypes[0]).view(-1, 2, 1, 1),
            "mid_ab": torch.stack(prototypes[1]).view(-1, 2, 1, 1),
            "highlight_ab": torch.stack(prototypes[2]).view(-1, 2, 1, 1),
            "global_ab": torch.stack(global_ab).view(-1, 2, 1, 1),
            "palette_reliability": torch.stack(reliability).view(-1, 1, 1, 1),
            "reference_hair_count": counts.view(-1, 1, 1, 1),
            "reference_lab": lab,
        }
        result["palette_shadow_rgb"] = lab_to_rgb(
            torch.cat((torch.full_like(result["shadow_ab"][:, :1], 50.0), result["shadow_ab"]), dim=1)
        )
        result["palette_mid_rgb"] = lab_to_rgb(
            torch.cat((torch.full_like(result["mid_ab"][:, :1], 50.0), result["mid_ab"]), dim=1)
        )
        result["palette_highlight_rgb"] = lab_to_rgb(
            torch.cat((torch.full_like(result["highlight_ab"][:, :1], 50.0), result["highlight_ab"]), dim=1)
        )
        return result


__all__ = ["ReferenceHairChromaPaletteV837"]
