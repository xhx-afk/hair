"""V2.44 carrier-preserving photometric residual correction.

The input carrier is the owner of local L/AB detail.  This module only edits
low-frequency tone/chroma and bounded illumination residuals.
"""

from __future__ import annotations

import torch

from models.gamut_precondition_v843 import gamut_precondition_v843
from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841


def _blur(value: torch.Tensor, radius: int = 11) -> torch.Tensor:
    import torch.nn.functional as F
    radius = max(int(radius), 1)
    return F.avg_pool2d(value, 2 * radius + 1, stride=1, padding=radius)


def _normalize(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(eps)


def _clamp_vector(value: torch.Tensor, maximum: float) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    scale = (float(maximum) / magnitude.clamp_min(1e-4)).clamp(max=1.0)
    return value * scale


class HairPhotometricResidualV844:
    def __init__(self, *, low_radius: int = 11, max_low_l_delta: float = 12.0,
                 low_freq_hue_mix: float = 0.15, max_c_delta: float = 10.0,
                 low_c_gain: float = 0.35, scene_ab_max: float = 5.0,
                 realism_enabled: bool = True) -> None:
        self.low_radius = int(low_radius)
        self.max_low_l_delta = float(max_low_l_delta)
        self.low_freq_hue_mix = float(low_freq_hue_mix)
        self.max_c_delta = float(max_c_delta)
        self.low_c_gain = float(low_c_gain)
        self.scene_ab_max = float(scene_ab_max)
        self.realism_enabled = bool(realism_enabled)

    @staticmethod
    def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
        x = ((value - low) / max(high - low, 1e-6)).clamp(0, 1)
        return x * x * (3.0 - 2.0 * x)

    def __call__(self, *, carrier_lab: torch.Tensor, desired_l_low: torch.Tensor,
                 reference_ab_low: torch.Tensor, scene_illumination_ab: torch.Tensor | None = None,
                 reference_chroma_low: torch.Tensor | None = None,
                 source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        carrier_l, carrier_ab = carrier_lab[:, :1], carrier_lab[:, 1:]
        carrier_l_low = _blur(carrier_l, self.low_radius)
        carrier_l_detail = carrier_l - carrier_l_low
        delta_l_low = (desired_l_low.float() - carrier_l_low).clamp(
            -self.max_low_l_delta, self.max_low_l_delta
        )
        final_l = carrier_l + delta_l_low

        carrier_ab_low = _blur(carrier_ab, self.low_radius)
        carrier_ab_detail = carrier_ab - carrier_ab_low
        carrier_c = torch.linalg.vector_norm(carrier_ab, dim=1, keepdim=True)
        carrier_c_low = torch.linalg.vector_norm(carrier_ab_low, dim=1, keepdim=True)
        reference_c = torch.linalg.vector_norm(reference_ab_low, dim=1, keepdim=True)
        reference_u = _normalize(reference_ab_low)
        carrier_u_low = _normalize(carrier_ab_low)
        direction_mix = self.low_freq_hue_mix
        mixed_u = _normalize((1.0 - direction_mix) * carrier_u_low + direction_mix * reference_u)
        target_c_low = reference_c if reference_chroma_low is None else reference_chroma_low.float().clamp_min(0.0)
        delta_c_low = (target_c_low - carrier_c_low).clamp(-self.max_c_delta, self.max_c_delta)

        shadow_scale = 0.30 + 0.70 * self._smoothstep(final_l, 12.0, 35.0)
        highlight_scale = 1.0 - 0.45 * self._smoothstep(final_l, 70.0, 92.0)
        plausibility = torch.ones_like(final_l)
        bleach_demand = torch.zeros_like(final_l)
        if self.realism_enabled and source_hair_l is not None and reference_l is not None:
            bleach_demand = (reference_l.float() - source_hair_l.float()).clamp_min(0.0)
            plausibility = 1.0 - 0.20 * (bleach_demand / 40.0).clamp(0, 1)

        final_c = carrier_c * shadow_scale * highlight_scale * plausibility
        final_c = (final_c + self.low_c_gain * delta_c_low).clamp_min(0.0)
        # Keep the carrier's local detail as an additive high-frequency owner.
        final_ab = mixed_u * final_c + carrier_ab_detail

        scene = torch.zeros_like(final_ab) if scene_illumination_ab is None else scene_illumination_ab.float()
        if scene.shape[-2:] != final_l.shape[-2:]:
            scene = scene.expand(-1, -1, final_l.shape[-2], final_l.shape[-1])
        scene_delta_ab = _clamp_vector(0.15 * scene, self.scene_ab_max)
        final_ab = final_ab + scene_delta_ab
        raw_lab = torch.cat((final_l, final_ab), dim=1)
        conditioned_lab, pre_scale = gamut_precondition_v843(raw_lab)
        corrected_rgb, gamut_aux = gamut_safe_lab_to_rgb_v841(conditioned_lab, return_aux=True)
        final_ab_safe = gamut_aux["gamut_safe_lab"][:, 1:]
        final_l_safe = gamut_aux["gamut_safe_l"]
        final_l_low = _blur(final_l_safe, self.low_radius)
        final_l_detail = final_l_safe - final_l_low
        final_ab_low = _blur(final_ab_safe, self.low_radius)
        final_ab_detail = final_ab_safe - final_ab_low
        total_scale = (pre_scale * gamut_aux["gamut_scale_map"]).clamp(0, 1)
        aux = {
            "carrier_lab": carrier_lab,
            "carrier_l": carrier_l,
            "carrier_l_low": carrier_l_low,
            "carrier_l_detail": carrier_l_detail,
            "desired_l_low": desired_l_low,
            "delta_l_low": delta_l_low,
            "final_l": final_l_safe,
            "final_l_low": final_l_low,
            "final_l_detail": final_l_detail,
            "carrier_ab": carrier_ab,
            "carrier_ab_low": carrier_ab_low,
            "carrier_ab_detail": carrier_ab_detail,
            "final_ab_low": final_ab_low,
            "final_ab_detail": final_ab_detail,
            "target_ab_low": reference_ab_low,
            "delta_ab_low": mixed_u * (self.low_c_gain * delta_c_low),
            "scene_delta_ab": scene_delta_ab,
            "final_ab": final_ab_safe,
            "carrier_chroma": carrier_c,
            "carrier_chroma_low": carrier_c_low,
            "photometric_chroma_scale": shadow_scale * highlight_scale * plausibility,
            "final_chroma": torch.linalg.vector_norm(final_ab_safe, dim=1, keepdim=True),
            "shadow_chroma_scale": shadow_scale,
            "highlight_chroma_scale": highlight_scale,
            "bleach_demand": bleach_demand,
            "plausibility_scale": plausibility,
            "pre_gamut_scale": pre_scale,
            "final_gamut_scale": gamut_aux["gamut_scale_map"],
            "total_gamut_scale": total_scale,
            "corrected_hair_rgb": corrected_rgb,
            **gamut_aux,
        }
        return (corrected_rgb, aux) if return_aux else corrected_rgb


__all__ = ["HairPhotometricResidualV844"]
