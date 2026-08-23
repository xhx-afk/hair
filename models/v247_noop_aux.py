"""Independent carrier baseline AUX for V2.47 attribution."""

from __future__ import annotations

import torch


def build_carrier_noop_aux(*, carrier_rgb: torch.Tensor, carrier_l: torch.Tensor,
                           carrier_l_low: torch.Tensor, carrier_ab: torch.Tensor,
                           carrier_ab_low: torch.Tensor, carrier_ab_detail: torch.Tensor,
                           target_hair_soft: torch.Tensor, stats_mask: torch.Tensor,
                           stats_mask_source: str, gate_metric_valid: torch.Tensor) -> dict[str, object]:
    batch = carrier_rgb.size(0)
    zeros = lambda value: torch.zeros_like(value)
    ones = lambda value: torch.ones_like(value)
    return {
        "carrier_l": carrier_l.clone(), "carrier_l_low": carrier_l_low.clone(),
        "carrier_ab": carrier_ab.clone(), "carrier_ab_low": carrier_ab_low.clone(),
        "carrier_ab_detail": carrier_ab_detail.clone(), "final_l": carrier_l.clone(),
        "final_ab": carrier_ab.clone(), "final_ab_low": carrier_ab_low.clone(),
        "provisional_ab": carrier_ab.clone(), "candidate_rgb": carrier_rgb.clone(),
        "target_hair_soft": target_hair_soft.clone(), "hair_apply_mask": target_hair_soft.clone(),
        "stats_mask": stats_mask.clone(), "stats_mask_source": stats_mask_source,
        "stats_mask_pixel_count": stats_mask.flatten(1).gt(.5).sum(1),
        "gate_metric_valid": gate_metric_valid.clone(),
        "l_gate_strength": zeros(gate_metric_valid), "ab_gate_strength": zeros(gate_metric_valid),
        "shadow_gate_strength": zeros(gate_metric_valid), "highlight_gate_strength": zeros(gate_metric_valid),
        "shading_gate_strength": zeros(gate_metric_valid), "plausibility_gate_strength": zeros(gate_metric_valid),
        "delta_l": zeros(carrier_l), "gated_delta_l": zeros(carrier_l),
        "delta_ab_center": zeros(carrier_ab), "l_delta_clamp_fraction": zeros(gate_metric_valid),
        "l_delta_p90": zeros(gate_metric_valid), "l_delta_mean_abs": zeros(gate_metric_valid),
        "ab_correction_magnitude": zeros(gate_metric_valid), "shadow_chroma_scale": ones(carrier_l),
        "highlight_chroma_scale": ones(carrier_l), "plausibility_scale": ones(carrier_l),
        "pre_gamut_scale": ones(carrier_l), "final_gamut_scale": ones(carrier_l),
        "total_gamut_scale": ones(carrier_l), "no_op": torch.ones(batch, device=carrier_rgb.device),
        "scene_tint_enabled": torch.tensor(False, device=carrier_rgb.device),
    }
