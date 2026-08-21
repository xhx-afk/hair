"""V2.40 multi-band texture-preserving recolor with V2.38 matte ownership."""

from __future__ import annotations

import torch

from models.face_contact_guard_v840 import FaceContactContaminationGuardV840
from models.hair_local_recomposition_v837 import HairLocalRecompositionV837
from models.hair_multiband_carrier_v840 import HairMultiBandCarrierV840
from models.reference_hair_intrinsic_tone_v839 import ReferenceHairTonePaletteV839
from models.reference_tone_mapper_v840 import ReferenceToneMapperV840
from models.target_hair_matte_refiner_v838 import TargetHairMatteRefinerV838
from models.target_hair_occlusion_resolver_v838 import TargetHairOcclusionResolverV838
from models.target_relative_chroma_field_v837 import TargetRelativeChromaFieldV837
from models.texture_preserving_recolor_v840 import TexturePreservingRecolorV840


class HairLocalRecompositionV840:
    def __init__(self, *, palette_mad_scale: float = 3.5, palette_min_support: int = 16,
                 illumination_radius: int = 11, carrier_mid_radius: int = 3,
                 mid_gain: float = 1.0, high_gain: float = 0.95,
                 chroma_mid_gain: float = 0.35, chroma_high_gain: float = 0.20,
                 contact_radius: int = 5, risk_strength: float = 0.65,
                 matte_ring_radius: int = 5, face_guard: float = 0.35,
                 contact_guard_strength: float = 0.85) -> None:
        self.palette = ReferenceHairTonePaletteV839(mad_scale=palette_mad_scale, min_support=palette_min_support)
        self.carrier = HairMultiBandCarrierV840(low_radius=illumination_radius, mid_radius=carrier_mid_radius)
        self.tone_mapper = ReferenceToneMapperV840()
        self.relative_field = TargetRelativeChromaFieldV837(low_radius=illumination_radius)
        self.recolor = TexturePreservingRecolorV840(
            mid_gain=mid_gain, high_gain=high_gain,
            chroma_mid_gain=chroma_mid_gain, chroma_high_gain=chroma_high_gain,
        )
        self.resolver = TargetHairOcclusionResolverV838(contact_radius=contact_radius, risk_strength=risk_strength)
        self.matte = TargetHairMatteRefinerV838(
            ring_radius=matte_ring_radius, face_guard=face_guard,
            use_appearance_confidence=False,
        )
        self.guard = FaceContactContaminationGuardV840(strength=contact_guard_strength)
        self.recomposer = HairLocalRecompositionV837(face_guard=0.0)

    def config_dict(self) -> dict[str, object]:
        return {
            "version": "v2.40",
            "carrier": "STRONG_ANCHOR_MULTIBAND",
            "mid_frequency_owner": "HAIR_CARRIER_PRESERVED",
            "high_frequency_owner": "HAIR_CARRIER_PRESERVED",
            "reference_tone_owner": "REFERENCE_TONE_QUANTILE_MAPPING",
            "chroma_owner": "REFERENCE_HUE_CARRIER_CHROMA_TEXTURE",
            "matte_owner": "V2.38_OCCLUSION_AWARE_MATTE",
            "face_contact_guard": "GEOMETRY_AND_CARRIER_EVIDENCE_ONLY",
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
        del target_illumination_rgb
        palette = self.palette(reference_rgb=color_reference_rgb, reference_hair_mask=reference_hair_mask)
        _, carrier_aux = self.carrier(
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb,
            allowed_hair_mask=allowed, hair_alpha_prior=allowed, return_aux=True,
        )
        mapped_l_low, tone_aux = self.tone_mapper(
            target_l_low=carrier_aux["carrier_l_low"], hair_mask=allowed,
            reference_tone=palette, return_aux=True,
        )
        field = self.relative_field(
            target_illumination_rgb=strong_anchor_rgb, target_hair_mask=allowed,
            target_hair_alpha=allowed, palette=palette,
        )
        new_hair_rgb, recolor_aux = self.recolor(
            carrier_aux=carrier_aux, mapped_l_low=mapped_l_low,
            target_ab_low=field["target_ab"], return_aux=True,
        )
        alpha, matte_aux = self.matte(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            new_hair_rgb=new_hair_rgb, coarse_target_hair_mask=coarse_target_hair_mask,
            occlusion_allowed_hair=allowed, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask,
            face_contact_ring=ownership_aux["face_contact_ring"],
            face_intrusion_risk=ownership_aux["face_intrusion_risk"], return_aux=True,
        )
        guarded_alpha, guard_aux = self.guard(
            alpha=alpha, allowed_hair=allowed, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask,
            face_contact_ring=ownership_aux["face_contact_ring"],
            face_intrusion_risk=ownership_aux["face_intrusion_risk"],
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb,
            new_hair_rgb=new_hair_rgb, hair_core=matte_aux["target_hair_core"],
            return_aux=True,
        )
        final_rgb = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=new_hair_rgb, hair_alpha=guarded_alpha,
            face_mask=torch.zeros_like(source_face_mask), target_hair_mask=torch.ones_like(guarded_alpha),
        )
        if not return_aux:
            return final_rgb
        aux = {
            **palette, **carrier_aux, **tone_aux, **field, **recolor_aux,
            **ownership_aux, **matte_aux, **guard_aux,
            "hair_ownership": allowed,
            "target_hair_alpha_final": guarded_alpha,
            "new_hair_rgb": new_hair_rgb, "final_rgb": final_rgb,
            "final_minus_base": final_rgb - base_rgb,
            "non_hair_mask": (1.0 - guarded_alpha).clamp(0, 1),
            "non_hair_leakage_map": (final_rgb - base_rgb).abs().mean(1, keepdim=True) * (1.0 - guarded_alpha),
            "face_rgb_change_from_base": ((final_rgb - base_rgb) * source_face_mask).abs().amax(dim=(1, 2, 3)),
            "non_hair_rgb_change_from_base": ((final_rgb - base_rgb) * (1.0 - guarded_alpha)).abs().amax(dim=(1, 2, 3)),
            "hair_undertransfer_map": matte_aux["target_hair_core"] * (recolor_aux["gamut_clip_map"] > 0).float(),
            # Compatibility diagnostic: report how much of the reference hue
            # magnitude is retained after carrier-texture recomposition.
            "chroma_strength": (recolor_aux["new_chroma"] / recolor_aux["reference_chroma_low"].clamp_min(1e-4)).clamp(0, 2),
        }
        return final_rgb, aux


__all__ = ["HairLocalRecompositionV840"]
