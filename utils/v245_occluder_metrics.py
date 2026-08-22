"""Occluder ownership metrics for V2.45 diagnostics."""

from __future__ import annotations

import torch

from utils.v240_metrics import _masked_mean


def occluder_metric_tensors(*, base_rgb: torch.Tensor, current_rgb: torch.Tensor,
                            protected_rgb: torch.Tensor, occluder_alpha: torch.Tensor,
                            hair_alpha: torch.Tensor, hair_alpha_reference: torch.Tensor,
                            hair_mask: torch.Tensor, real_mask: torch.Tensor,
                            synthetic_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    occ = occluder_alpha.float().clamp(0, 1)
    outside = (1.0 - occ).clamp(0, 1)
    edge = (torch.nn.functional.max_pool2d(occ, 3, stride=1, padding=1) - torch.nn.functional.max_pool2d(-occ, 3, stride=1, padding=1)).clamp(0, 1)
    out = {
        "occluder_rgb_change_from_base": _masked_mean((protected_rgb - base_rgb).abs().mean(1, keepdim=True), occ),
        "occluder_p90_change": torch.quantile(((protected_rgb - base_rgb).abs().mean(1, keepdim=True) * occ).flatten(1), 0.90, dim=1),
        "occluder_max_change": ((protected_rgb - base_rgb).abs().mean(1, keepdim=True) * occ).flatten(1).amax(1),
        "occluder_edge_color_bleed": _masked_mean((protected_rgb - base_rgb).abs().mean(1, keepdim=True), edge),
        "hair_coverage_outside_occluder": _masked_mean(hair_alpha, hair_mask * outside),
        "hair_alpha_change_outside_occluder": _masked_mean((hair_alpha - hair_alpha_reference).abs(), hair_mask * outside),
        "real_occluder_pixel_count": real_mask.flatten(1).sum(1),
        "synthetic_occluder_pixel_count": synthetic_mask.flatten(1).sum(1),
    }
    return {key: torch.nan_to_num(value) for key, value in out.items()}


def classify_occluder(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"decision": "OCCLUDER_NO_VALID_SAMPLES", "count": 0, "key_metrics": {}}
    med = lambda key: float(torch.tensor([float(row.get(key, 0.0)) for row in records]).median())
    p90 = float(torch.tensor([float(row.get("occluder_p90_change", 0.0)) for row in records]).quantile(0.90))
    if med("occluder_rgb_change_from_base") <= 0.005 and p90 <= 0.02 and med("hair_alpha_change_outside_occluder") <= 0.02:
        decision = "EXPLICIT_OCCLUDER_OWNERSHIP_REQUIRED"
    elif med("synthetic_occluder_pixel_count") > 0 and med("real_occluder_pixel_count") <= 0:
        decision = "OCCLUDER_MODEL_COVERAGE_INSUFFICIENT"
    else:
        decision = "OCCLUDER_PROTECTION_INCONCLUSIVE"
    return {"decision": decision, "count": len(records), "key_metrics": {"median_occluder_rgb_change": med("occluder_rgb_change_from_base"), "p90_occluder_change": p90, "median_hair_alpha_change_outside": med("hair_alpha_change_outside_occluder")}}


__all__ = ["classify_occluder", "occluder_metric_tensors"]
