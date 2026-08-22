"""V2.47 error-aware, carrier-first appearance residual."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.carrier_reference_error_v847 import CarrierReferenceErrorEstimatorV847
from models.gamut_precondition_v843 import gamut_precondition_v843
from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841
from models.reference_ab_statistics_v847 import reference_ab_statistics
from models.v245_death_test_common import dilate
from models.SG_IDCT_v16 import rgb_to_lab


def _blur(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    return F.avg_pool2d(value.float(), 2 * radius + 1, stride=1, padding=radius)


def _normalize(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(eps)


def _clamp_vector(value: torch.Tensor, maximum: float) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return value * (float(maximum) / magnitude.clamp_min(1e-4)).clamp(max=1.0)


def _region_ratio(chroma: torch.Tensor, luma: torch.Tensor, mask: torch.Tensor, low: float, high: float | None = None) -> torch.Tensor:
    q_low = torch.quantile(luma.flatten(1), luma.new_tensor(low), dim=1).view(-1, 1, 1, 1)
    region = mask * ((luma <= q_low) if high is None else ((luma >= q_low) & (luma <= torch.quantile(luma.flatten(1), luma.new_tensor(high), dim=1).view(-1, 1, 1, 1)))).float()
    return (chroma * region).flatten(1).sum(1) / region.flatten(1).sum(1).clamp_min(1.0)


def _masked_q(value: torch.Tensor, mask: torch.Tensor, level: float) -> torch.Tensor:
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > .5]
        rows.append(torch.quantile(pixels, value.new_tensor(level)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


class HairPhotometricResidualV847:
    scene_tint_enabled = False
    v247_plausibility_enabled = False

    def __init__(self, *, low_radius: int = 11, residual_radius: int = 5,
                 max_delta_l: float = 5.0, max_delta_ab: float = 6.0,
                 plausibility_enabled: bool = False) -> None:
        self.low_radius, self.residual_radius = int(low_radius), int(residual_radius)
        self.max_delta_l, self.max_delta_ab = float(max_delta_l), float(max_delta_ab)
        self.plausibility_enabled = bool(plausibility_enabled)
        self.estimator = CarrierReferenceErrorEstimatorV847()

    def __call__(self, *, carrier_rgb: torch.Tensor, reference_rgb: torch.Tensor,
                 target_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor | None = None,
                 source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None,
                 enable_l: bool = True, enable_ab: bool = True,
                 enable_shading: bool = False, enable_plausibility: bool = False,
                 return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        carrier_lab = rgb_to_lab(carrier_rgb)
        carrier_l, carrier_ab = carrier_lab[:, :1], carrier_lab[:, 1:]
        target_soft = _blur(target_hair_mask.float().clamp(0, 1), 5).clamp(0, 1)
        hair_apply = target_soft * (target_soft >= 0.01).float()
        error = self.estimator(carrier_rgb=carrier_rgb, reference_rgb=reference_rgb, carrier_hair_mask=target_hair_mask, reference_hair_mask=reference_hair_mask)
        carrier_l_low = _blur(carrier_l, self.low_radius)
        carrier_center = error["carrier_l_q50"].view(-1, 1, 1, 1)
        reference_center = error["reference_l_q50"].view(-1, 1, 1, 1)
        carrier_iqr = (error["carrier_l_q75"] - error["carrier_l_q25"]).clamp_min(3.0)
        reference_iqr = (error["reference_l_q75"] - error["reference_l_q25"]).clamp_min(3.0)
        l_scale = (reference_iqr / carrier_iqr).clamp(.8, 1.2)
        desired_l_low = reference_center + l_scale.view(-1, 1, 1, 1) * (carrier_l_low - carrier_center)
        delta_l_raw = desired_l_low - carrier_l_low
        delta_l = _blur(delta_l_raw, self.residual_radius).clamp(-self.max_delta_l, self.max_delta_l)
        l_gate = error["l_gate_strength"].view(-1, 1, 1, 1) if enable_l else torch.zeros_like(target_soft)
        gated_delta_l = l_gate * delta_l * target_soft
        final_l = carrier_l + gated_delta_l

        carrier_ab_low = _blur(carrier_ab, self.low_radius)
        carrier_ab_detail = carrier_ab - carrier_ab_low
        carrier_ab_center = error["carrier_median_ab"].view(-1, 2, 1, 1)
        reference_ab_center = error["reference_median_ab"].view(-1, 2, 1, 1)
        delta_ab_center = _clamp_vector(reference_ab_center - carrier_ab_center, self.max_delta_ab)
        ab_gate = error["ab_gate_strength"].view(-1, 1, 1, 1) if enable_ab else torch.zeros_like(target_soft)
        final_ab_low = carrier_ab_low + ab_gate * delta_ab_center * target_soft
        hue_mix = torch.where((error["hue_error_deg"] > 5.0) & (error["reference_chroma_median"] >= 12.0), (ab_gate.flatten() * .15).clamp(max=.15), torch.zeros_like(ab_gate.flatten())).view(-1, 1, 1, 1)
        final_ab_low = _normalize((1.0 - hue_mix) * final_ab_low + hue_mix * reference_ab_center)
        final_ab_low = final_ab_low * torch.linalg.vector_norm(carrier_ab_low + ab_gate * delta_ab_center * target_soft, dim=1, keepdim=True).clamp_min(0.0)
        provisional_ab = final_ab_low + carrier_ab_detail
        provisional_c = provisional_ab.norm(dim=1, keepdim=True)

        carrier_shadow = _region_ratio(carrier_ab.norm(dim=1, keepdim=True), carrier_l, target_hair_mask, .20)
        carrier_mid = _region_ratio(carrier_ab.norm(dim=1, keepdim=True), carrier_l, target_hair_mask, .35, .65)
        carrier_high = (carrier_ab.norm(dim=1, keepdim=True) * target_hair_mask * (carrier_l >= _masked_q(carrier_l, target_hair_mask, .80).view(-1, 1, 1, 1)).float()).flatten(1).sum(1) / target_hair_mask.flatten(1).sum(1).clamp_min(1.0)
        shadow_over = (carrier_shadow / carrier_mid.clamp_min(1e-4) - .90).clamp_min(0.0)
        highlight_over = (carrier_high / carrier_mid.clamp_min(1e-4) - 1.0).clamp_min(0.0)
        shading_gate = torch.maximum((shadow_over / .30).clamp(0, 1), (highlight_over / .30).clamp(0, 1)) if enable_shading else torch.zeros_like(shadow_over)
        shading_gate = shading_gate.view(-1, 1, 1, 1)
        q20 = _masked_q(final_l, target_hair_mask, .20).view(-1, 1, 1, 1); q35 = _masked_q(final_l, target_hair_mask, .35).view(-1, 1, 1, 1); q65 = _masked_q(final_l, target_hair_mask, .65).view(-1, 1, 1, 1); q80 = _masked_q(final_l, target_hair_mask, .80).view(-1, 1, 1, 1)
        rank = ((final_l - q20) / (q80 - q20).clamp_min(1e-4)).clamp(0, 1)
        shadow_raw = .75 + .25 * ((rank - .10) / .25).clamp(0, 1)
        highlight_raw = 1.0 - .20 * ((rank - .75) / .20).clamp(0, 1)
        shadow_scale = 1.0 + shading_gate * (shadow_raw - 1.0) * target_soft
        highlight_scale = 1.0 + shading_gate * (highlight_raw - 1.0) * target_soft
        final_c = provisional_c * shadow_scale * highlight_scale
        plausibility_gate = torch.zeros_like(shading_gate)
        plausibility = torch.ones_like(final_l)
        if enable_plausibility and self.plausibility_enabled and source_hair_l is not None and reference_l is not None:
            plausibility_gate = torch.full_like(shading_gate, .5)
            plausibility = 1.0 - .20 * ((reference_l.float() - source_hair_l.float()).clamp_min(0.0) / 40.0).clamp(0, 1) * target_soft
            final_c = final_c * plausibility
        final_ab = _normalize(provisional_ab) * final_c
        conditioned, pre_scale = gamut_precondition_v843(torch.cat((final_l, final_ab), dim=1))
        rgb_safe, gamut_aux = gamut_safe_lab_to_rgb_v841(conditioned, return_aux=True)
        # The candidate is hair-only: restore the frozen carrier everywhere
        # outside the soft target ownership map after gamut conversion.
        candidate_rgb = (carrier_rgb + hair_apply * (rgb_safe - carrier_rgb)).clamp(0, 1)
        candidate_rgb = torch.where(error["no_op"].view(-1, 1, 1, 1) > 0.5, carrier_rgb, candidate_rgb)
        l_pixels = []
        for index in range(delta_l.size(0)):
            pixels = delta_l[index].abs().flatten()[target_soft[index].flatten() > .5]
            l_pixels.append(torch.quantile(pixels, delta_l.new_tensor(.90)) if pixels.numel() else delta_l.new_zeros(()))
        l_delta_p90 = torch.stack(l_pixels)
        aux = {**error, "carrier_l": carrier_l, "carrier_l_low": carrier_l_low, "desired_l_low": desired_l_low, "delta_l_low_raw": delta_l_raw, "delta_l": delta_l, "gated_delta_l": gated_delta_l, "final_l": final_l, "carrier_ab": carrier_ab, "carrier_ab_low": carrier_ab_low, "carrier_ab_detail": carrier_ab_detail, "delta_ab_center": delta_ab_center, "final_ab_low": final_ab_low, "provisional_ab": provisional_ab, "final_ab": conditioned[:, 1:], "shadow_chroma_scale": shadow_scale, "highlight_chroma_scale": highlight_scale, "shading_gate_strength": shading_gate, "plausibility_scale": plausibility, "plausibility_gate_strength": plausibility_gate, "target_hair_soft": target_soft, "hair_apply_mask": hair_apply, "l_delta_clamp_fraction": (delta_l.abs() >= self.max_delta_l - 1e-5).float().flatten(1).sum(1) / target_soft.flatten(1).sum(1).clamp_min(1.0), "l_delta_p90": l_delta_p90, "l_delta_mean_abs": (delta_l.abs() * target_soft).flatten(1).sum(1) / target_soft.flatten(1).sum(1).clamp_min(1.0), "ab_correction_magnitude": delta_ab_center.norm(dim=1).flatten(), "total_gamut_scale": (pre_scale * gamut_aux["gamut_scale_map"]).clamp(0, 1), "pre_gamut_scale": pre_scale, "final_gamut_scale": gamut_aux["gamut_scale_map"], "candidate_rgb": candidate_rgb, "scene_tint_enabled": torch.tensor(False, device=carrier_rgb.device), **gamut_aux}
        return (candidate_rgb, aux) if return_aux else candidate_rgb


__all__ = ["HairPhotometricResidualV847"]
