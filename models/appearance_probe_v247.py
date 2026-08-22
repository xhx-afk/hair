"""Component-attribution Appearance Probe for V2.47."""

from __future__ import annotations

import torch

from models.hair_photometric_residual_v847 import HairPhotometricResidualV847
from models.v245_death_test_common import composite, erode, smoothstep


class AppearanceProbeV247:
    def __init__(self) -> None:
        self.photometric = HairPhotometricResidualV847()
        self.photometric_with_plausibility = HairPhotometricResidualV847(plausibility_enabled=True)

    def trusted_core(self, target_hair_mask: torch.Tensor, anchor_hair_evidence: torch.Tensor) -> torch.Tensor:
        return erode(target_hair_mask.float().clamp(0, 1), 6) * smoothstep(anchor_hair_evidence.float(), .45, .70)

    def __call__(self, *, carrier_rgb: torch.Tensor, reference_rgb: torch.Tensor,
                 target_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor, source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None, trusted_alpha: torch.Tensor | None = None,
                 return_aux: bool = False):
        common = dict(carrier_rgb=carrier_rgb, reference_rgb=reference_rgb, target_hair_mask=target_hair_mask, reference_hair_mask=reference_hair_mask, strong_anchor_rgb=strong_anchor_rgb, source_hair_l=source_hair_l, reference_l=reference_l, carrier_stats_mask=trusted_alpha)
        candidates = {}
        aux = {}
        for name, options in (("c1", dict(enable_l=True, enable_ab=False)), ("c2", dict(enable_l=False, enable_ab=True)), ("c3", dict(enable_l=True, enable_ab=True)), ("c4", dict(enable_l=True, enable_ab=True, enable_shading=True))):
            candidates[name], aux[name] = self.photometric(**common, **options, return_aux=True)
        candidates["c5"], aux["c5"] = self.photometric_with_plausibility(**common, enable_l=True, enable_ab=True, enable_shading=True, enable_plausibility=True, return_aux=True)
        # C0 is a real carrier baseline.  Keep only geometry/statistics needed
        # for diagnostics; every correction-related field must describe a no-op.
        candidates["c0"] = carrier_rgb
        aux["c0"] = dict(aux["c1"])
        carrier_lab = aux["c1"].get("carrier_l")
        carrier_ab = aux["c1"].get("carrier_ab")
        if carrier_lab is not None:
            aux["c0"]["final_l"] = carrier_lab.clone()
        if carrier_ab is not None:
            aux["c0"]["final_ab"] = carrier_ab.clone()
            aux["c0"]["final_ab_low"] = aux["c1"].get("carrier_ab_low", carrier_ab).clone()
            aux["c0"]["provisional_ab"] = carrier_ab.clone()
        for key in ("l_gate_strength", "ab_gate_strength", "shadow_gate_strength", "highlight_gate_strength", "shading_gate_strength", "plausibility_gate_strength", "l_delta", "gated_delta_l", "delta_ab_center", "l_delta_clamp_fraction", "l_delta_p90", "l_delta_mean_abs", "ab_correction_magnitude"):
            if key in aux["c0"] and torch.is_tensor(aux["c0"][key]):
                aux["c0"][key] = torch.zeros_like(aux["c0"][key])
        for key in ("shadow_chroma_scale", "highlight_chroma_scale", "plausibility_scale", "total_gamut_scale", "pre_gamut_scale", "final_gamut_scale"):
            if key in aux["c0"] and torch.is_tensor(aux["c0"][key]):
                aux["c0"][key] = torch.ones_like(aux["c0"][key])
        aux["c0"]["candidate_rgb"] = carrier_rgb.clone()
        aux["c0"]["no_op"] = torch.ones_like(aux["c1"].get("no_op", torch.ones(carrier_rgb.size(0), device=carrier_rgb.device)))
        outputs = {f"{name}_rgb": value for name, value in candidates.items()}
        if trusted_alpha is not None:
            outputs.update({name.replace("_rgb", "_preview"): composite(carrier_rgb, value, trusted_alpha) for name, value in outputs.items()})
        return (outputs, aux) if return_aux else outputs

    def run_selected(self, *, enable_l: bool, enable_ab: bool, enable_shading: bool,
                     enable_plausibility: bool, **kwargs):
        engine = self.photometric_with_plausibility if enable_plausibility else self.photometric
        return engine(enable_l=enable_l, enable_ab=enable_ab, enable_shading=enable_shading, enable_plausibility=enable_plausibility, return_aux=True, **kwargs)


__all__ = ["AppearanceProbeV247"]
