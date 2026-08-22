"""Frozen-alpha appearance variants for the V2.45 Appearance Death Test."""

from __future__ import annotations

import torch

from models.hair_photometric_residual_v844 import HairPhotometricResidualV844
from models.v245_death_test_common import composite, erode


class AppearanceProbeV245:
    def __init__(self) -> None:
        self.photometric = HairPhotometricResidualV844()

    def trusted_core(self, coarse_target_hair_mask: torch.Tensor,
                     anchor_hair_evidence: torch.Tensor) -> torch.Tensor:
        return erode(coarse_target_hair_mask.float().clamp(0, 1), 8) * (anchor_hair_evidence.float() > 0.60).float()

    def __call__(self, *, carrier_lab: torch.Tensor, desired_l_low: torch.Tensor,
                 reference_ab_low: torch.Tensor, scene_illumination_ab: torch.Tensor,
                 source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None,
                 base_rgb: torch.Tensor | None = None,
                 trusted_alpha: torch.Tensor | None = None,
                 return_aux: bool = False):
        carrier_rgb, current_aux = self.photometric(carrier_lab=carrier_lab, desired_l_low=desired_l_low,
                                       reference_ab_low=reference_ab_low, scene_illumination_ab=scene_illumination_ab,
                                       source_hair_l=source_hair_l, reference_l=reference_l, return_aux=True)
        no_scene_rgb, no_scene_aux = self.photometric(carrier_lab=carrier_lab, desired_l_low=desired_l_low,
            reference_ab_low=reference_ab_low, scene_illumination_ab=torch.zeros_like(scene_illumination_ab),
            source_hair_l=source_hair_l, reference_l=reference_l, return_aux=True)
        no_shading_rgb, no_shading_aux = self.photometric(carrier_lab=carrier_lab, desired_l_low=desired_l_low,
            reference_ab_low=reference_ab_low, scene_illumination_ab=scene_illumination_ab,
            source_hair_l=source_hair_l, reference_l=reference_l, disable_shading=True, return_aux=True)
        outputs = {"a0_carrier_rgb": None, "a1_photometric_rgb": carrier_rgb,
                   "a2_no_scene_rgb": no_scene_rgb, "a3_no_shading_rgb": no_shading_rgb}
        if base_rgb is not None and trusted_alpha is not None:
            outputs.update({key.replace("_rgb", "_preview"): composite(base_rgb, value, trusted_alpha)
                            for key, value in outputs.items() if value is not None})
        aux = {"a1": current_aux, "a2": no_scene_aux, "a3": no_shading_aux}
        return (outputs, aux) if return_aux else outputs


__all__ = ["AppearanceProbeV245"]
