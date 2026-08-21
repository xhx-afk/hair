"""Multi-band texture-preserving hair recolor."""

from __future__ import annotations

import torch

from models.gamut_safe_lab_v840 import gamut_safe_lab_to_rgb_v840


class TexturePreservingRecolorV840:
    def __init__(self, *, mid_gain: float = 1.0, high_gain: float = 0.95,
                 chroma_mid_gain: float = 0.35, chroma_high_gain: float = 0.20,
                 detail_limit: float = 30.0, max_chroma: float = 55.0) -> None:
        if not 0.85 <= mid_gain <= 1.10 or not 0.80 <= high_gain <= 1.10:
            raise ValueError("V2.40 luminance gains are outside the contract")
        self.mid_gain, self.high_gain = float(mid_gain), float(high_gain)
        self.chroma_mid_gain, self.chroma_high_gain = float(chroma_mid_gain), float(chroma_high_gain)
        self.detail_limit, self.max_chroma = float(detail_limit), float(max_chroma)

    def __call__(self, *, carrier_aux: dict[str, torch.Tensor], mapped_l_low: torch.Tensor,
                 target_ab_low: torch.Tensor, return_aux: bool = False):
        l_detail = (self.mid_gain * carrier_aux["carrier_l_mid"] + self.high_gain * carrier_aux["carrier_l_high"]).clamp(-self.detail_limit, self.detail_limit)
        final_l = mapped_l_low + l_detail
        ref_chroma = torch.linalg.vector_norm(target_ab_low, dim=1, keepdim=True)
        u_ref = target_ab_low / ref_chroma.clamp_min(1e-4)
        c_new = (ref_chroma + self.chroma_mid_gain * carrier_aux["carrier_chroma_mid"] + self.chroma_high_gain * carrier_aux["carrier_chroma_high"]).clamp(0, self.max_chroma)
        low_chroma = ref_chroma < 4.0
        c_new = torch.where(low_chroma, ref_chroma + 0.25 * self.chroma_mid_gain * carrier_aux["carrier_chroma_mid"] + 0.10 * self.chroma_high_gain * carrier_aux["carrier_chroma_high"], c_new).clamp(0, self.max_chroma)
        final_ab = u_ref * c_new
        # If the reference hue is undefined, keep only a very small neutral
        # texture contribution and avoid importing the carrier's original hue.
        final_ab = torch.where(low_chroma, final_ab * 0.25, final_ab)
        lab = torch.cat((final_l, final_ab), dim=1)
        rgb, gamut_scale, gamut_clip = gamut_safe_lab_to_rgb_v840(lab)
        if not return_aux:
            return rgb
        return rgb, {
            "recolor_lab": lab,
            "new_hair_rgb": rgb,
            "final_l": final_l,
            "l_detail": l_detail,
            "target_ab_low": target_ab_low,
            "reference_chroma_low": ref_chroma,
            "new_chroma": c_new,
            "gamut_chroma_scale": gamut_scale,
            "gamut_clip_map": gamut_clip,
        }


__all__ = ["TexturePreservingRecolorV840"]
