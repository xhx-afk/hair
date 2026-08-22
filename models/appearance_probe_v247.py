"""Component-attribution Appearance Probe for V2.47."""

from __future__ import annotations

import torch

from models.hair_photometric_residual_v847 import HairPhotometricResidualV847
from models.v245_death_test_common import composite, erode, smoothstep


class AppearanceProbeV247:
    def __init__(self) -> None:
        self.photometric = HairPhotometricResidualV847()

    def trusted_core(self, target_hair_mask: torch.Tensor, anchor_hair_evidence: torch.Tensor) -> torch.Tensor:
        return erode(target_hair_mask.float().clamp(0, 1), 6) * smoothstep(anchor_hair_evidence.float(), .45, .70)

    def __call__(self, *, carrier_rgb: torch.Tensor, reference_rgb: torch.Tensor,
                 target_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor, source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None, trusted_alpha: torch.Tensor | None = None,
                 return_aux: bool = False):
        common = dict(carrier_rgb=carrier_rgb, reference_rgb=reference_rgb, target_hair_mask=target_hair_mask, reference_hair_mask=reference_hair_mask, strong_anchor_rgb=strong_anchor_rgb, source_hair_l=source_hair_l, reference_l=reference_l)
        candidates = {}
        aux = {}
        for name, options in (("c0", dict()), ("c1", dict(enable_l=True, enable_ab=False)), ("c2", dict(enable_l=False, enable_ab=True)), ("c3", dict(enable_l=True, enable_ab=True)), ("c4", dict(enable_l=True, enable_ab=True, enable_shading=True)), ("c5", dict(enable_l=True, enable_ab=True, enable_shading=True, enable_plausibility=True))):
            candidates[name], aux[name] = self.photometric(**common, **options, return_aux=True)
        candidates["c0"] = carrier_rgb
        outputs = {f"{name}_rgb": value for name, value in candidates.items()}
        if trusted_alpha is not None:
            outputs.update({name.replace("_rgb", "_preview"): composite(carrier_rgb, value, trusted_alpha) for name, value in outputs.items()})
        return (outputs, aux) if return_aux else outputs


__all__ = ["AppearanceProbeV247"]
