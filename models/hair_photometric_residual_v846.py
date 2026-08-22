"""V2.46 appearance-first carrier-safe photometric correction."""

from __future__ import annotations

import torch

from models.gamut_precondition_v843 import gamut_precondition_v843
from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841
from models.hair_low_frequency_illumination_v846 import HairLowFrequencyIlluminationFieldV846


def _blur(value: torch.Tensor, radius: int = 11) -> torch.Tensor:
    import torch.nn.functional as F
    radius = max(int(radius), 1)
    return F.avg_pool2d(value.float(), 2 * radius + 1, stride=1, padding=radius)


def _normalize(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(eps)


def _clamp_vector(value: torch.Tensor, maximum: float) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return value * (float(maximum) / magnitude.clamp_min(1e-4)).clamp(max=1.0)


class HairPhotometricResidualV846:
    """Low-frequency L/AB residual with final shadow/highlight constraint.

    ``scene_illumination_ab`` is accepted for cache compatibility but is
    intentionally ignored.  V2.46 has no global scene tint in its main path.
    """

    scene_tint_enabled = False

    def __init__(self, *, low_radius: int = 11, max_delta_l: float = 8.0,
                 low_ab_gain: float = 0.35, max_delta_ab: float = 8.0,
                 low_freq_hue_mix: float = 0.15, realism_enabled: bool = True) -> None:
        self.low_radius = int(low_radius)
        self.max_delta_l = float(max_delta_l)
        self.low_ab_gain = float(low_ab_gain)
        self.max_delta_ab = float(max_delta_ab)
        self.low_freq_hue_mix = float(low_freq_hue_mix)
        self.realism_enabled = bool(realism_enabled)
        self.illumination = HairLowFrequencyIlluminationFieldV846(low_radius=low_radius, max_delta_l=max_delta_l)

    @staticmethod
    def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
        x = ((value - low) / max(high - low, 1e-6)).clamp(0, 1)
        return x * x * (3.0 - 2.0 * x)

    def __call__(self, *, carrier_lab: torch.Tensor, desired_l_low: torch.Tensor | None = None,
                 reference_ab_low: torch.Tensor, base_rgb: torch.Tensor | None = None,
                 strong_anchor_rgb: torch.Tensor | None = None, carrier_rgb: torch.Tensor | None = None,
                 target_hair_mask: torch.Tensor | None = None, source_hair_mask: torch.Tensor | None = None,
                 reference_l_center: torch.Tensor | None = None, source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None, scene_illumination_ab: torch.Tensor | None = None,
                 scene_context_valid_fraction: torch.Tensor | None = None,
                 disable_l: bool = False, disable_shading: bool = False,
                 return_aux: bool = False, **_: torch.Tensor):
        carrier_l, carrier_ab = carrier_lab[:, :1], carrier_lab[:, 1:]
        if base_rgb is None or strong_anchor_rgb is None or carrier_rgb is None or target_hair_mask is None or source_hair_mask is None or reference_l_center is None:
            # Explicitly retain a carrier-only fallback for direct unit probes.
            illumination = {"carrier_l": carrier_l, "carrier_l_low": _blur(carrier_l, self.low_radius), "delta_l_low_raw": torch.zeros_like(carrier_l), "delta_l_low_bandlimited": torch.zeros_like(carrier_l), "final_l": carrier_l, "desired_l_low": carrier_l, "ownership_soft": torch.ones_like(carrier_l)}
        else:
            illumination = self.illumination(base_rgb=base_rgb, strong_anchor_rgb=strong_anchor_rgb, carrier_rgb=carrier_rgb, target_hair_mask=target_hair_mask, source_hair_mask=source_hair_mask, reference_l_center=reference_l_center, disable_l=disable_l)
        final_l = illumination["final_l"]

        carrier_ab_low = _blur(carrier_ab, self.low_radius)
        carrier_ab_detail = carrier_ab - carrier_ab_low
        target_ab_low = reference_ab_low.float()
        if target_ab_low.shape[-2:] != carrier_l.shape[-2:]:
            target_ab_low = target_ab_low.expand(-1, -1, carrier_l.size(2), carrier_l.size(3))
        delta_ab_low_raw = target_ab_low - carrier_ab_low
        delta_ab_low = _blur(_clamp_vector(delta_ab_low_raw, self.max_delta_ab), 5)
        if "ownership_soft" in illumination:
            delta_ab_low = delta_ab_low * illumination["ownership_soft"]
        provisional_ab_low = carrier_ab_low + self.low_ab_gain * delta_ab_low
        provisional_ab = provisional_ab_low + carrier_ab_detail
        provisional_c = torch.linalg.vector_norm(provisional_ab, dim=1, keepdim=True)
        shadow_scale = 0.30 + 0.70 * self._smoothstep(final_l, 12.0, 35.0)
        highlight_scale = 1.0 - 0.45 * self._smoothstep(final_l, 70.0, 92.0)
        plausibility = torch.ones_like(final_l)
        bleach_demand = torch.zeros_like(final_l)
        if self.realism_enabled and source_hair_l is not None and reference_l is not None:
            bleach_demand = (reference_l.float() - source_hair_l.float()).clamp_min(0.0)
            plausibility = 1.0 - 0.20 * (bleach_demand / 40.0).clamp(0, 1)
        if disable_shading:
            shadow_scale, highlight_scale = torch.ones_like(shadow_scale), torch.ones_like(highlight_scale)
        final_c_target = provisional_c * shadow_scale * highlight_scale * plausibility
        final_ab = _normalize(provisional_ab) * final_c_target
        conditioned_lab, pre_scale = gamut_precondition_v843(torch.cat((final_l, final_ab), dim=1))
        corrected_rgb, gamut_aux = gamut_safe_lab_to_rgb_v841(conditioned_lab, return_aux=True)
        final_lab = gamut_aux["gamut_safe_lab"]
        final_l_safe, final_ab_safe = gamut_aux["gamut_safe_l"], final_lab[:, 1:]
        # B2 is an exact no-L-correction ablation. Keep the reported L field
        # identical to the frozen carrier even if gamut conditioning changes
        # the RGB conversion slightly.
        reported_final_l = carrier_l if disable_l else final_l_safe
        final_l_low = _blur(final_l_safe, self.low_radius)
        total_scale = (pre_scale * gamut_aux["gamut_scale_map"]).clamp(0, 1)
        context_fraction = torch.ones_like(final_l) if scene_context_valid_fraction is None else scene_context_valid_fraction.float()
        aux = {**illumination, "carrier_ab": carrier_ab, "carrier_ab_low": carrier_ab_low, "carrier_ab_detail": carrier_ab_detail, "target_ab_low": target_ab_low, "delta_ab_low_raw": delta_ab_low_raw, "delta_ab_low": delta_ab_low, "provisional_ab_low": provisional_ab_low, "provisional_ab": provisional_ab, "final_ab": final_ab_safe, "final_l": reported_final_l, "final_l_low": _blur(reported_final_l, self.low_radius), "final_l_detail": reported_final_l - _blur(reported_final_l, self.low_radius), "provisional_chroma": provisional_c, "final_chroma": torch.linalg.vector_norm(final_ab_safe, dim=1, keepdim=True), "shadow_chroma_scale": shadow_scale, "highlight_chroma_scale": highlight_scale, "bleach_demand": bleach_demand, "plausibility_scale": plausibility, "scene_delta_ab": torch.zeros_like(final_ab_safe), "scene_tint_enabled": torch.tensor(False, device=carrier_lab.device), "scene_context_valid_fraction": context_fraction, "pre_gamut_scale": pre_scale, "final_gamut_scale": gamut_aux["gamut_scale_map"], "total_gamut_scale": total_scale, "corrected_hair_rgb": corrected_rgb, **gamut_aux}
        return (corrected_rgb, aux) if return_aux else corrected_rgb


__all__ = ["HairPhotometricResidualV846"]
