"""Decompose the real Strong Anchor hair carrier into texture bands."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class HairMultiBandCarrierV840:
    def __init__(self, *, low_radius: int = 11, mid_radius: int = 3) -> None:
        if low_radius <= mid_radius or mid_radius < 1:
            raise ValueError("V2.40 carrier radii are invalid")
        self.low_radius = int(low_radius)
        self.mid_radius = int(mid_radius)

    def __call__(self, *, strong_anchor_rgb: torch.Tensor, base_rgb: torch.Tensor,
                 allowed_hair_mask: torch.Tensor, hair_alpha_prior: torch.Tensor | None = None,
                 return_aux: bool = False):
        del base_rgb
        carrier_lab = rgb_to_lab(strong_anchor_rgb)
        l = carrier_lab[:, :1]
        l_low = gaussian_blur_v829(l, self.low_radius)
        l_mid0 = gaussian_blur_v829(l, self.mid_radius)
        l_mid = l_mid0 - l_low
        l_high = l - l_mid0
        ab = carrier_lab[:, 1:]
        chroma = torch.linalg.vector_norm(ab, dim=1, keepdim=True)
        c_low = gaussian_blur_v829(chroma, self.low_radius)
        c_mid0 = gaussian_blur_v829(chroma, self.mid_radius)
        c_mid = c_mid0 - c_low
        c_high = chroma - c_mid0
        mask = allowed_hair_mask.float().clamp(0, 1)
        alpha = mask if hair_alpha_prior is None else hair_alpha_prior.float().clamp(0, 1)
        aux = {
            "carrier_lab": carrier_lab,
            "carrier_l": l,
            "carrier_ab": ab,
            "carrier_l_low": l_low,
            "carrier_l_mid": l_mid,
            "carrier_l_high": l_high,
            "carrier_chroma": chroma,
            "carrier_chroma_low": c_low,
            "carrier_chroma_mid": c_mid,
            "carrier_chroma_high": c_high,
            "carrier_hair_mask": mask,
            "carrier_hair_alpha_prior": alpha,
        }
        return (carrier_lab, aux) if return_aux else carrier_lab


__all__ = ["HairMultiBandCarrierV840"]
