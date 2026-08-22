"""V2.43 confidence-core/strand-matte upgrade over the V2.42 transfer path.

The established appearance transfer remains the source of the color carrier;
only matte ownership and skin protection are replaced with confidence-aware
operators.  This keeps V2.42's coverage and multi-band texture contract while
making the new behavior opt-in and easy to compare.
"""

from __future__ import annotations

import torch

from models.face_hair_matte_guard_v843 import FaceHairMatteGuardV843
from models.hair_core_confidence_v843 import HairCoreConfidenceV843
from models.hair_local_recomposition_v842 import HairLocalRecompositionV842
from models.hair_photometric_appearance_v843 import HairPhotometricAppearanceV843
from models.gamut_precondition_v843 import gamut_precondition_v843
from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841
from models.new_hair_illumination_v843 import new_hair_illumination_v843
from models.scene_illumination_estimator_v843 import SceneIlluminationEstimatorV843
from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.strand_aware_hair_matte_v843 import StrandAwareHairMatteV843


class HairLocalRecompositionV843(HairLocalRecompositionV842):
    def __init__(self, **kwargs) -> None:
        kwargs["version"] = "v2.43"
        super().__init__(**kwargs)
        self.core_confidence = HairCoreConfidenceV843(
            distance_steps=kwargs.get("confidence_distance_steps", 12),
            safe_threshold=kwargs.get("safe_core_threshold", 0.72),
        )
        self.strand_matte = StrandAwareHairMatteV843(
            support_radius=kwargs.get("flyaway_support_radius", 4),
            safe_alpha=kwargs.get("safe_core_alpha", 0.92),
        )
        self.v243_guard = FaceHairMatteGuardV843(
            strength=kwargs.get("contact_guard_strength", 0.85)
        )
        self.scene_illumination = SceneIlluminationEstimatorV843(scene_mix=0.15)
        self.photometric = HairPhotometricAppearanceV843(
            scene_mix=0.15, realism_enabled=kwargs.get("photometric_realism_enabled", True)
        )

    def config_dict(self) -> dict[str, object]:
        config = super().config_dict()
        config.update({
            "version": "v2.43",
            "ownership_owner": "DISTANCE_PRIOR_ANCHOR_EVIDENCE_STRAND_CONFIDENCE",
            "matte_owner": "STRAND_AWARE_CONFIDENCE_CORE",
            "flyaway_support": "DILATED_TARGET_SUPPORT_WITH_DUAL_THRESHOLD",
            "appearance_owner": "STABLE_HUE_PHOTOMETRIC_COMPATIBLE",
            "v243_photometric_realism_enabled": True,
        })
        return config

    def __call__(self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 source_hair_mask: torch.Tensor | None = None, return_aux: bool = False):
        # Run the proven appearance path first; it supplies the stable-hue and
        # strong-anchor carrier that V2.43 explicitly preserves.
        _, old_aux = super().__call__(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            color_reference_rgb=color_reference_rgb, target_illumination_rgb=target_illumination_rgb,
            coarse_target_hair_mask=coarse_target_hair_mask, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask, reference_hair_mask=reference_hair_mask,
            source_hair_mask=source_hair_mask, return_aux=True,
        )
        # Couple stable reference hue to target scene light.  The old carrier
        # supplies L and strand texture; this stage only adjusts chroma and is
        # bounded by the same local gamut solver used by V2.42.
        old_lab = rgb_to_lab(old_aux["new_hair_rgb"])
        illumination_ownership = new_hair_illumination_v843(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            target_hair_mask=coarse_target_hair_mask, source_hair_mask=source_hair_mask,
            reference_l_center=old_aux["l_q50"], radius=11,
        )
        old_lab = torch.cat((illumination_ownership["mapped_l_low"], old_lab[:, 1:]), dim=1)
        base_l = rgb_to_lab(base_rgb)[:, :1]
        source_mask = torch.zeros_like(base_l) if source_hair_mask is None else source_hair_mask.float().clamp(0, 1)
        source_l_values = []
        for batch_index in range(base_l.size(0)):
            pixels = base_l[batch_index, 0][source_mask[batch_index, 0] > 0.5]
            source_l_values.append(pixels.median() if pixels.numel() else base_l[batch_index, 0].median())
        source_hair_l = torch.stack(source_l_values).view(-1, 1, 1, 1)
        scene_aux = self.scene_illumination(
            target_rgb=target_illumination_rgb,
            target_hair_mask=coarse_target_hair_mask,
            return_aux=True,
        )
        photometric_ab, photometric_aux = self.photometric(
            reference_ab=old_aux.get("target_ab_low", old_lab[:, 1:]),
            final_l=old_lab[:, :1],
            scene_illumination_ab=scene_aux["scene_illumination_ab"],
            source_hair_l=source_hair_l,
            reference_l=old_aux.get("l_q50"),
            return_aux=True,
        )
        conditioned_lab, precondition_scale = gamut_precondition_v843(
            torch.cat((old_lab[:, :1], photometric_ab), dim=1)
        )
        photometric_rgb, gamut_aux = gamut_safe_lab_to_rgb_v841(conditioned_lab, return_aux=True)
        evidence = old_aux["anchor_hair_evidence"]
        confidence = self.core_confidence(
            coarse_target_hair_mask=coarse_target_hair_mask,
            source_hair_mask=source_hair_mask,
            source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb,
            anchor_hair_evidence=evidence,
            return_aux=True,
        )
        alpha, matte_aux = self.strand_matte(
            coarse_target_hair_mask=coarse_target_hair_mask,
            safe_dense_core=confidence["safe_dense_core"],
            uncertain_core=confidence["uncertain_core"],
            transition_prior=confidence["transition_prior"],
            anchor_hair_evidence=evidence,
            strand_structure_confidence=confidence["strand_structure_confidence"],
            source_skin_mask=source_skin_mask,
            skin_conflict=confidence["skin_conflict"],
            return_aux=True,
        )
        alpha, guard_aux = self.v243_guard(
            alpha=alpha, source_skin_mask=source_skin_mask,
            dense_core_confidence=confidence["dense_core_confidence"],
            strand_structure_confidence=confidence["strand_structure_confidence"],
            anchor_hair_evidence=evidence, return_aux=True,
        )
        final_rgb = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=photometric_rgb, hair_alpha=alpha,
            face_mask=torch.zeros_like(source_face_mask), target_hair_mask=torch.ones_like(alpha),
        )
        if not return_aux:
            return final_rgb
        old_aux.update(confidence)
        old_aux.update(matte_aux)
        old_aux.update(guard_aux)
        old_aux.update(scene_aux)
        old_aux.update(illumination_ownership)
        old_aux.update(photometric_aux)
        old_aux.update({"gamut_precondition_scale": precondition_scale, **gamut_aux})
        old_aux.update({
            "target_hair_alpha_final": alpha,
            "new_hair_rgb": photometric_rgb,
            "final_rgb": final_rgb,
            "v243_photometric_realism_enabled": True,
        })
        return final_rgb, old_aux


__all__ = ["HairLocalRecompositionV843"]
