"""B0/B1/B2/B3 appearance variants for the V2.46 death test."""

from __future__ import annotations

import torch

from models.hair_photometric_residual_v846 import HairPhotometricResidualV846
from models.v245_death_test_common import composite, erode, smoothstep


class AppearanceProbeV246:
    def __init__(self) -> None:
        self.photometric = HairPhotometricResidualV846()

    def trusted_core(self, target_hair_mask: torch.Tensor, anchor_hair_evidence: torch.Tensor) -> torch.Tensor:
        return erode(target_hair_mask.float().clamp(0, 1), 6) * smoothstep(anchor_hair_evidence.float(), 0.45, 0.70)

    def __call__(self, *, carrier_lab: torch.Tensor, base_rgb: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor, carrier_rgb: torch.Tensor,
                 target_hair_mask: torch.Tensor, source_hair_mask: torch.Tensor,
                 reference_l_center: torch.Tensor, reference_ab_low: torch.Tensor,
                 source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None,
                 scene_context_valid_fraction: torch.Tensor | None = None,
                 trusted_alpha: torch.Tensor | None = None,
                 return_aux: bool = False):
        common = dict(carrier_lab=carrier_lab, base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb, carrier_rgb=carrier_rgb, target_hair_mask=target_hair_mask, source_hair_mask=source_hair_mask, reference_l_center=reference_l_center, reference_ab_low=reference_ab_low, source_hair_l=source_hair_l, reference_l=reference_l, scene_illumination_ab=None, scene_context_valid_fraction=scene_context_valid_fraction)
        b1, aux1 = self.photometric(**common, return_aux=True)
        b2, aux2 = self.photometric(**common, disable_l=True, return_aux=True)
        b3, aux3 = self.photometric(**common, disable_shading=True, return_aux=True)
        outputs = {"b0_carrier_rgb": carrier_rgb, "b1_full_rgb": b1, "b2_no_l_rgb": b2, "b3_no_shading_rgb": b3}
        if trusted_alpha is not None:
            outputs.update({name.replace("_rgb", "_preview"): composite(base_rgb, rgb, trusted_alpha) for name, rgb in outputs.items()})
        return (outputs, {"b1": aux1, "b2": aux2, "b3": aux3}) if return_aux else outputs


__all__ = ["AppearanceProbeV246"]
