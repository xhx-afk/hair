"""V2.38 local appearance recomposition using refined occlusion-aware alpha."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hair_local_appearance_recomposition_v837 import HairLocalAppearanceRecompositionV837
from models.hair_local_recomposition_v837 import HairLocalRecompositionV837
from models.target_hair_matte_refiner_v838 import TargetHairMatteRefinerV838
from models.target_hair_occlusion_resolver_v838 import TargetHairOcclusionResolverV838


class HairLocalRecompositionV838:
    """Keep v2.37 palette/L structure while repairing ownership and final matte."""

    def __init__(self, *, palette_mad_scale: float = 3.5, palette_min_support: int = 16,
                 illumination_radius: int = 11, anchor_hf_gain: float = 0.9,
                 contact_radius: int = 5, risk_strength: float = 0.65,
                 matte_ring_radius: int = 5, face_guard: float = 0.35) -> None:
        self.appearance = HairLocalAppearanceRecompositionV837(
            palette_mad_scale=palette_mad_scale, palette_min_support=palette_min_support,
            illumination_radius=illumination_radius, anchor_hf_gain=anchor_hf_gain,
            face_guard=0.0,
        )
        self.resolver = TargetHairOcclusionResolverV838(
            contact_radius=contact_radius, risk_strength=risk_strength
        )
        self.matte = TargetHairMatteRefinerV838(
            ring_radius=matte_ring_radius, face_guard=face_guard
        )
        self.recomposer = HairLocalRecompositionV837(face_guard=0.0)

    def config_dict(self) -> dict[str, object]:
        return {
            "version": "v2.38",
            "non_hair_owner": "BASE_SOURCE_PRESERVED",
            "topology_owner": "TARGET_HAIR_COARSE",
            "occlusion_owner": "TARGET_HAIR_OCCLUSION_RESOLVER",
            "matte_owner": "HIGH_RES_RULE_BASED_REFINER",
            "source_face_role": "SOFT_CONTACT_GUIDANCE_ONLY",
        }

    def __call__(
        self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
        color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
        coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
        source_skin_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        allowed, ownership_aux = self.resolver(
            coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb, return_aux=True,
        )
        # Source face is intentionally zeroed here: resolver/matte own occlusion.
        _, appearance_aux = self.appearance(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            color_reference_rgb=color_reference_rgb, target_illumination_rgb=target_illumination_rgb,
            target_hair_mask=allowed, face_mask=torch.zeros_like(source_face_mask),
            reference_hair_mask=reference_hair_mask, target_hair_alpha=allowed,
            return_aux=True,
        )
        alpha, matte_aux = self.matte(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            new_hair_rgb=appearance_aux["new_hair_rgb"],
            coarse_target_hair_mask=coarse_target_hair_mask,
            occlusion_allowed_hair=allowed, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask,
            face_contact_ring=ownership_aux["face_contact_ring"],
            face_intrusion_risk=ownership_aux["face_intrusion_risk"], return_aux=True,
        )
        final_rgb = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=appearance_aux["new_hair_rgb"],
            hair_alpha=alpha, face_mask=torch.zeros_like(source_face_mask),
            target_hair_mask=torch.ones_like(alpha), return_aux=False,
        )
        if not return_aux:
            return final_rgb
        final_aux = {
            **appearance_aux, **ownership_aux, **matte_aux,
            "hair_ownership": allowed,
            "target_hair_alpha_final": alpha,
            "final_rgb": final_rgb,
            "final_minus_base": final_rgb - base_rgb,
            "non_hair_mask": (1.0 - alpha).clamp(0, 1),
            "non_hair_leakage_map": (final_rgb - base_rgb).abs().mean(1, keepdim=True) * (1.0 - alpha),
            "face_rgb_change_from_base": ((final_rgb - base_rgb) * source_face_mask).abs().amax(dim=(1, 2, 3)),
            "non_hair_rgb_change_from_base": ((final_rgb - base_rgb) * (1.0 - alpha)).abs().amax(dim=(1, 2, 3)),
        }
        # Correct diagnostic: completed chroma transfer relative to Base, not a self ratio.
        base_ab = rgb_to_lab(base_rgb)[:, 1:]
        final_ab = rgb_to_lab(final_rgb)[:, 1:]
        target_ab = appearance_aux["target_ab"]
        final_aux["chroma_strength"] = (
            torch.linalg.vector_norm(final_ab - base_ab, dim=1, keepdim=True)
            / torch.linalg.vector_norm(target_ab - base_ab, dim=1, keepdim=True).clamp_min(1e-4)
        ).clamp(0, 1)
        # Undertransfer must be measured on the final matte core, not the
        # pre-refinement v2.37 ownership core.
        final_aux["hair_undertransfer_map"] = matte_aux["target_hair_core"] * (
            final_aux["chroma_strength"] < 0.5
        ).float()
        return final_rgb, final_aux


__all__ = ["HairLocalRecompositionV838"]
