"""V2.39 intrinsic-tone hair recomposition on top of V2.38 matte ownership."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hair_local_recomposition_v837 import HairLocalRecompositionV837
from models.hair_tone_recomposer_v839 import HairToneRecomposerV839
from models.reference_hair_intrinsic_tone_v839 import ReferenceHairTonePaletteV839
from models.target_hair_matte_refiner_v838 import TargetHairMatteRefinerV838
from models.target_hair_occlusion_resolver_v838 import TargetHairOcclusionResolverV838
from models.target_illumination_normalizer_v839 import TargetIlluminationNormalizerV839
from models.target_relative_chroma_field_v837 import TargetRelativeChromaFieldV837
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class HairLocalRecompositionV839:
    def __init__(self, *, palette_mad_scale: float = 3.5, palette_min_support: int = 16,
                 illumination_radius: int = 11, illumination_scale: float = 0.9,
                 anchor_hf_gain: float = 0.9, contact_radius: int = 5,
                 risk_strength: float = 0.65, matte_ring_radius: int = 5,
                 face_guard: float = 0.35) -> None:
        self.palette = ReferenceHairTonePaletteV839(mad_scale=palette_mad_scale, min_support=palette_min_support)
        self.relative_field = TargetRelativeChromaFieldV837(low_radius=illumination_radius)
        self.illumination = TargetIlluminationNormalizerV839(
            low_radius=illumination_radius, illumination_scale=illumination_scale
        )
        self.tone = HairToneRecomposerV839(
            illumination_scale=illumination_scale, anchor_hf_gain=anchor_hf_gain
        )
        self.resolver = TargetHairOcclusionResolverV838(
            contact_radius=contact_radius, risk_strength=risk_strength
        )
        self.matte = TargetHairMatteRefinerV838(ring_radius=matte_ring_radius, face_guard=face_guard)
        self.recomposer = HairLocalRecompositionV837(face_guard=0.0)

    def config_dict(self) -> dict[str, object]:
        return {
            "version": "v2.39",
            "non_hair_owner": "BASE_SOURCE_PRESERVED",
            "topology_owner": "V2.38_OCCLUSION_AWARE_MATTE",
            "reference_tone_owner": "REFERENCE_HAIR_INTRINSIC_L",
            "target_illumination_owner": "TARGET_LOW_FREQUENCY_RESIDUAL",
            "source_face_role": "SOFT_CONTACT_GUIDANCE_ONLY",
        }

    def __call__(self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 return_aux: bool = False):
        allowed, ownership_aux = self.resolver(
            coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb, return_aux=True,
        )
        palette = self.palette(reference_rgb=color_reference_rgb, reference_hair_mask=reference_hair_mask)
        field = self.relative_field(
            target_illumination_rgb=target_illumination_rgb,
            target_hair_mask=allowed, target_hair_alpha=allowed, palette=palette,
        )
        illum = self.illumination(
            target_illumination_rgb=target_illumination_rgb, target_hair_mask=allowed
        )
        new_hair_rgb, tone_aux = self.tone(
            palette=palette, target_l_low=illum["target_l_low"],
            target_l_center=illum["target_l_center"],
            target_illumination_residual=illum["target_illumination_residual"],
            anchor_rgb=strong_anchor_rgb, target_ab=field["target_ab"],
        )
        alpha, matte_aux = self.matte(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            new_hair_rgb=new_hair_rgb, coarse_target_hair_mask=coarse_target_hair_mask,
            occlusion_allowed_hair=allowed, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask,
            face_contact_ring=ownership_aux["face_contact_ring"],
            face_intrusion_risk=ownership_aux["face_intrusion_risk"], return_aux=True,
        )
        final_rgb = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=new_hair_rgb, hair_alpha=alpha,
            face_mask=torch.zeros_like(source_face_mask), target_hair_mask=torch.ones_like(alpha),
        )
        if not return_aux:
            return final_rgb
        base_ab = rgb_to_lab(base_rgb)[:, 1:]
        final_ab = rgb_to_lab(final_rgb)[:, 1:]
        final_aux = {
            **palette, **field, **illum, **tone_aux, **ownership_aux, **matte_aux,
            "hair_ownership": allowed, "target_hair_alpha_final": alpha,
            "new_hair_rgb": new_hair_rgb, "final_rgb": final_rgb,
            "final_minus_base": final_rgb - base_rgb,
            "non_hair_mask": (1.0 - alpha).clamp(0, 1),
            "non_hair_leakage_map": (final_rgb - base_rgb).abs().mean(1, keepdim=True) * (1.0 - alpha),
            "face_rgb_change_from_base": ((final_rgb - base_rgb) * source_face_mask).abs().amax(dim=(1, 2, 3)),
            "non_hair_rgb_change_from_base": ((final_rgb - base_rgb) * (1.0 - alpha)).abs().amax(dim=(1, 2, 3)),
        }
        final_aux["chroma_strength"] = (
            torch.linalg.vector_norm(final_ab - base_ab, dim=1, keepdim=True)
            / torch.linalg.vector_norm(field["target_ab"] - base_ab, dim=1, keepdim=True).clamp_min(1e-4)
        ).clamp(0, 1)
        final_aux["hair_undertransfer_map"] = matte_aux["target_hair_core"] * (
            final_aux["chroma_strength"] < 0.5
        ).float()
        return final_rgb, final_aux


__all__ = ["HairLocalRecompositionV839"]
