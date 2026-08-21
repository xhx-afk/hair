"""Intrinsic-tone plus target-illumination LAB recomposition with gamut safety."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import lab_to_rgb
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class HairToneRecomposerV839:
    def __init__(self, *, illumination_scale: float = 0.9, anchor_hf_gain: float = 0.9, gamut_clip_ratio: float = 0.005) -> None:
        if not 0.0 <= illumination_scale <= 1.5 or not 0.0 <= anchor_hf_gain <= 1.5:
            raise ValueError("V2.39 tone recomposer parameters are invalid")
        self.illumination_scale = float(illumination_scale)
        self.anchor_hf_gain = float(anchor_hf_gain)
        self.gamut_clip_ratio = float(gamut_clip_ratio)

    @staticmethod
    def _tone_weight(chroma: torch.Tensor) -> torch.Tensor:
        # Low-chroma brown/gray/black hair needs the reference intrinsic tone;
        # saturated colors already have a strong AB identity.
        return torch.where(
            chroma <= 8.0, torch.ones_like(chroma),
            torch.where(chroma >= 30.0, torch.full_like(chroma, 0.60), 1.0 - 0.4 * (chroma - 8.0) / 22.0),
        ).clamp(0.5, 1.0)

    def _gamut_safe(self, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = lab_to_rgb(lab)
        clip_ratio = ((rgb < 0.0) | (rgb > 1.0)).float().mean(dim=(1, 2, 3), keepdim=True)
        scales = (1.0, 0.95, 0.90, 0.85, 0.80)
        chosen = torch.full_like(clip_ratio, scales[-1])
        safe_rgb = rgb.clone()
        safe_clip = clip_ratio
        for scale in scales:
            candidate = lab_to_rgb(torch.cat((lab[:, :1], lab[:, 1:] * scale), dim=1))
            candidate_clip = ((candidate < 0.0) | (candidate > 1.0)).float().mean(dim=(1, 2, 3), keepdim=True)
            use = (safe_clip > self.gamut_clip_ratio) & (candidate_clip <= self.gamut_clip_ratio)
            safe_rgb = torch.where(use, candidate, safe_rgb)
            chosen = torch.where(use, torch.full_like(chosen, scale), chosen)
            safe_clip = torch.where(use, candidate_clip, safe_clip)
        unresolved = safe_clip > self.gamut_clip_ratio
        fallback = lab_to_rgb(torch.cat((lab[:, :1], lab[:, 1:] * scales[-1]), dim=1))
        fallback_clip = ((fallback < 0.0) | (fallback > 1.0)).float().mean(dim=(1, 2, 3), keepdim=True)
        safe_rgb = torch.where(unresolved, fallback, safe_rgb)
        safe_clip = torch.where(unresolved, fallback_clip, safe_clip)
        return safe_rgb.clamp(0, 1), chosen, safe_clip

    def __call__(self, *, palette: dict[str, torch.Tensor], target_l_low: torch.Tensor,
                 target_l_center: torch.Tensor, target_illumination_residual: torch.Tensor,
                 anchor_rgb: torch.Tensor, target_ab: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from models.SG_IDCT_v16 import rgb_to_lab
        anchor_l = rgb_to_lab(anchor_rgb)[:, :1]
        anchor_low = gaussian_blur_v829(anchor_l, 3)
        anchor_hf = anchor_l - anchor_low
        chroma = torch.linalg.vector_norm(palette["global_ab"], dim=1, keepdim=True)
        tone_weight = self._tone_weight(chroma)
        ref_center = palette["intrinsic_l_center"]
        target_center = target_l_center
        final_center = tone_weight * ref_center + (1.0 - tone_weight) * target_center
        final_l_low = final_center + self.illumination_scale * target_illumination_residual
        final_l = final_l_low + self.anchor_hf_gain * anchor_hf
        lab = torch.cat((final_l, target_ab), dim=1)
        rgb, gamut_scale, clip_ratio = self._gamut_safe(lab)
        return rgb, {
            "target_l_low": target_l_low,
            "target_l_center": target_center,
            "final_l_center": final_center,
            "final_l_low": final_l_low,
            "final_l": final_l,
            "anchor_high_frequency_l": anchor_hf,
            "tone_weight": tone_weight,
            "gamut_chroma_scale": gamut_scale,
            "gamut_clip_ratio": clip_ratio,
            "tone_chroma": chroma,
        }


__all__ = ["HairToneRecomposerV839"]
