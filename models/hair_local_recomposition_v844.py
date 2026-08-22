"""V2.44 carrier-preserving residual photometric recomposition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.face_hair_matte_guard_v843 import FaceHairMatteGuardV843
from models.hair_core_confidence_v843 import HairCoreConfidenceV843
from models.hair_local_recomposition_v842 import HairLocalRecompositionV842
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844
from models.new_hair_illumination_v843 import new_hair_illumination_v843
from models.scene_illumination_estimator_v843 import SceneIlluminationEstimatorV843
from models.strand_aware_hair_matte_v843 import StrandAwareHairMatteV843


def _independent_flyaway_candidate(anchor_rgb: torch.Tensor, coarse: torch.Tensor,
                                   hair_support: torch.Tensor,
                                   strand: torch.Tensor) -> torch.Tensor:
    contrast = (anchor_rgb - F.avg_pool2d(anchor_rgb, 5, stride=1, padding=2)).abs().mean(1, keepdim=True)
    q90 = torch.quantile(contrast.flatten(1), 0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
    local_contrast = (contrast / q90).clamp(0, 1)
    outside = (hair_support * (1.0 - coarse)).clamp(0, 1)
    return outside * (strand > 0.55).float() * (local_contrast > 0.45).float()


class HairLocalRecompositionV844(HairLocalRecompositionV842):
    def __init__(self, **kwargs) -> None:
        kwargs["version"] = "v2.44"
        super().__init__(**kwargs)
        self.core_confidence = HairCoreConfidenceV843(
            distance_steps=kwargs.get("confidence_distance_steps", 12),
            safe_threshold=kwargs.get("safe_core_threshold", 0.72),
        )
        self.strand_matte = StrandAwareHairMatteV843(
            support_radius=kwargs.get("flyaway_support_radius", 4),
            safe_alpha=kwargs.get("safe_core_alpha", 0.92),
        )
        self.v244_guard = FaceHairMatteGuardV843(strength=kwargs.get("contact_guard_strength", 0.85))
        self.scene_illumination = SceneIlluminationEstimatorV843(scene_mix=0.15)
        self.photometric = HairPhotometricResidualV844(
            low_radius=11, max_low_l_delta=12.0, low_freq_hue_mix=0.15,
            max_c_delta=10.0, low_c_gain=0.35, scene_ab_max=5.0,
            realism_enabled=kwargs.get("photometric_realism_enabled", True),
        )

    def config_dict(self) -> dict[str, object]:
        config = super().config_dict()
        config.update({
            "version": "v2.44",
            "carrier_owner": "V2.42_NEW_HAIR_RGB",
            "l_owner": "CARRIER_L_PLUS_BOUNDED_LOW_FREQUENCY_RESIDUAL",
            "ab_owner": "CARRIER_AB_DETAIL_PLUS_LOW_FREQUENCY_RESIDUAL",
            "photometric_owner": "BOUNDED_RESIDUAL_CORRECTION",
            "candidate_owner": "HAIR_SUPPORT_ONLY",
            "v242_carrier_preservation_contract": True,
        })
        return config

    @staticmethod
    def _source_hair_l(base_rgb: torch.Tensor, source_hair_mask: torch.Tensor | None) -> torch.Tensor:
        base_l = rgb_to_lab(base_rgb)[:, :1]
        mask = torch.zeros_like(base_l) if source_hair_mask is None else source_hair_mask.float().clamp(0, 1)
        values = []
        for index in range(base_l.size(0)):
            pixels = base_l[index, 0][mask[index, 0] > 0.5]
            values.append(pixels.median() if pixels.numel() else base_l[index, 0].median())
        return torch.stack(values).view(-1, 1, 1, 1)

    def __call__(self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 source_hair_mask: torch.Tensor | None = None, return_aux: bool = False):
        # Bypass V2.43's full-frame Lab reconstruction and obtain the intact
        # V2.42 carrier/appearance path as the immutable local-detail source.
        _, carrier_aux = HairLocalRecompositionV842.__call__(
            self, base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            color_reference_rgb=color_reference_rgb, target_illumination_rgb=target_illumination_rgb,
            coarse_target_hair_mask=coarse_target_hair_mask, source_face_mask=source_face_mask,
            source_skin_mask=source_skin_mask, reference_hair_mask=reference_hair_mask,
            source_hair_mask=source_hair_mask, return_aux=True,
        )
        carrier_rgb = carrier_aux["new_hair_rgb"]
        carrier_lab = rgb_to_lab(carrier_rgb)
        ownership = new_hair_illumination_v843(
            base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb,
            target_hair_mask=coarse_target_hair_mask, source_hair_mask=source_hair_mask,
            reference_l_center=carrier_aux["l_q50"], radius=11,
            carrier_l=carrier_lab[:, :1], max_low_l_delta=12.0,
        )
        scene_aux = self.scene_illumination(
            target_rgb=target_illumination_rgb, target_hair_mask=coarse_target_hair_mask,
            return_aux=True,
        )
        corrected_rgb, residual_aux = self.photometric(
            carrier_lab=carrier_lab,
            desired_l_low=ownership["desired_l_low"],
            reference_ab_low=carrier_aux["target_ab_low"],
            reference_chroma_low=carrier_aux.get("reference_chroma_low"),
            scene_illumination_ab=scene_aux["scene_illumination_ab"],
            source_hair_l=self._source_hair_l(base_rgb, source_hair_mask),
            reference_l=carrier_aux.get("l_q50"),
            return_aux=True,
        )
        evidence = carrier_aux["anchor_hair_evidence"]
        confidence = self.core_confidence(
            coarse_target_hair_mask=coarse_target_hair_mask, source_hair_mask=source_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, anchor_hair_evidence=evidence, return_aux=True,
        )
        alpha, matte_aux = self.strand_matte(
            coarse_target_hair_mask=coarse_target_hair_mask,
            safe_dense_core=confidence["safe_dense_core"], uncertain_core=confidence["uncertain_core"],
            transition_prior=confidence["transition_prior"], anchor_hair_evidence=evidence,
            strand_structure_confidence=confidence["strand_structure_confidence"],
            source_skin_mask=source_skin_mask, skin_conflict=confidence["skin_conflict"], return_aux=True,
        )
        independent = _independent_flyaway_candidate(
            strong_anchor_rgb, coarse_target_hair_mask, matte_aux["hair_support"],
            confidence["strand_structure_confidence"],
        )
        alpha = torch.maximum(alpha, independent * 0.15).clamp(0, 1)
        alpha, guard_aux = self.v244_guard(
            alpha=alpha, source_skin_mask=source_skin_mask,
            dense_core_confidence=confidence["dense_core_confidence"],
            strand_structure_confidence=confidence["strand_structure_confidence"],
            anchor_hair_evidence=evidence, return_aux=True,
        )
        hair_support = matte_aux["hair_support"]
        candidate_rgb = hair_support * corrected_rgb + (1.0 - hair_support) * base_rgb
        final_rgb = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=candidate_rgb, hair_alpha=alpha,
            face_mask=torch.zeros_like(source_face_mask), target_hair_mask=torch.ones_like(alpha),
        )
        aux = {**carrier_aux, **ownership, **scene_aux, **residual_aux,
               **confidence, **matte_aux, **guard_aux,
               "carrier_rgb": carrier_rgb,
               "new_hair_rgb_v242_carrier": carrier_rgb,
               "new_hair_rgb": candidate_rgb,
               "hair_candidate_rgb": candidate_rgb,
               "independent_flyaway_candidate": independent,
               "independent_flyaway_recovery": (alpha > 0.1).float() * independent,
               "target_hair_alpha_final": alpha,
               "final_rgb": final_rgb,
               "final_minus_base": final_rgb - base_rgb,
               "v244_photometric_realism_enabled": True,
               "l_detail_loss_map": (residual_aux["final_l_detail"] - residual_aux["carrier_l_detail"]).abs(),
               "ab_detail_loss_map": torch.linalg.vector_norm(
                   (residual_aux["final_ab_detail"] - residual_aux["carrier_ab_detail"]), dim=1, keepdim=True
               ),
        }
        if not return_aux:
            return final_rgb
        return final_rgb, aux


__all__ = ["HairLocalRecompositionV844"]
