"""V2.41 texture-preserving recolor with stable hue and soft L compression."""

from __future__ import annotations

import torch

from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841
from models.reference_chroma_budget_v842 import apply_reference_chroma_budget_v842


def soft_clip_lab_l_v8411(value: torch.Tensor, *, low: float = 2.0,
                           high: float = 98.0, knee: float = 4.0) -> torch.Tensor:
    """Keep normal Lab L unchanged and roll off only out-of-range excess."""
    low_value = float(low) - float(knee) * torch.tanh((float(low) - value) / float(knee))
    high_value = float(high) + float(knee) * torch.tanh((value - float(high)) / float(knee))
    return torch.where(value < low, low_value, torch.where(value > high, high_value, value))


class TexturePreservingRecolorV841:
    def __init__(self, *, mid_gain: float = 1.0, high_gain: float = 0.95,
                 chroma_mid_gain: float = 0.35, chroma_high_gain: float = 0.20,
                 detail_limit: float = 30.0, max_chroma: float = 55.0) -> None:
        self.mid_gain, self.high_gain = float(mid_gain), float(high_gain)
        self.chroma_mid_gain, self.chroma_high_gain = float(chroma_mid_gain), float(chroma_high_gain)
        self.detail_limit, self.max_chroma = float(detail_limit), float(max_chroma)

    def __call__(self, *, carrier_aux: dict[str, torch.Tensor], mapped_l_low: torch.Tensor,
                 target_ab_low: torch.Tensor, hue_dispersion_deg: torch.Tensor | None = None,
                 return_aux: bool = False):
        stability = (1.0 - hue_dispersion_deg / 45.0).clamp(0, 1) if hue_dispersion_deg is not None else torch.ones_like(mapped_l_low)
        mid_gain = self.chroma_mid_gain * stability
        high_gain = self.chroma_high_gain * stability
        l_detail = (self.mid_gain * carrier_aux["carrier_l_mid"] + self.high_gain * carrier_aux["carrier_l_high"]).clamp(-self.detail_limit, self.detail_limit)
        final_l_raw = mapped_l_low + l_detail
        final_l_soft = soft_clip_lab_l_v8411(final_l_raw)
        ref_chroma = torch.linalg.vector_norm(target_ab_low, dim=1, keepdim=True)
        u_ref = target_ab_low / ref_chroma.clamp_min(1e-4)
        c_new = (ref_chroma + mid_gain * carrier_aux["carrier_chroma_mid"] + high_gain * carrier_aux["carrier_chroma_high"]).clamp(0, self.max_chroma)
        low_chroma = ref_chroma < 4.0
        c_new = torch.where(low_chroma, ref_chroma + 0.25 * mid_gain * carrier_aux["carrier_chroma_mid"] + 0.10 * high_gain * carrier_aux["carrier_chroma_high"], c_new).clamp(0, self.max_chroma)
        c_new, chroma_budget = apply_reference_chroma_budget_v842(
            c_new, lightness=final_l_soft, reference_chroma=ref_chroma
        )
        final_ab = u_ref * c_new
        lab = torch.cat((final_l_soft, final_ab), dim=1)
        rgb, gamut_aux = gamut_safe_lab_to_rgb_v841(lab, return_aux=True)
        final_l_safe = gamut_aux["gamut_safe_l"]
        if not return_aux:
            return rgb
        return rgb, {
            "recolor_lab": gamut_aux["gamut_safe_lab"], "new_hair_rgb": rgb,
            "final_l": final_l_safe, "final_l_raw": final_l_raw,
            "final_l_soft": final_l_soft, "final_l_safe": final_l_safe, "l_detail": l_detail,
            "target_ab_low": target_ab_low, "reference_chroma_low": ref_chroma,
            "new_chroma": c_new, "effective_chroma_mid_gain": mid_gain,
            "effective_chroma_high_gain": high_gain,
            "hue_stability": stability, **gamut_aux,
            "reference_chroma_budget": chroma_budget,
            "gamut_chroma_scale": gamut_aux["gamut_scale_map"],
            "gamut_clip_map": gamut_aux["gamut_compressed_map"],
        }


__all__ = ["TexturePreservingRecolorV841", "soft_clip_lab_l_v8411"]
