"""Metrics and decision logic for the V2.45 Matte Death Test."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab


def _mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().expand_as(value)
    return (value.float() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def _q(value: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        rows.append(torch.quantile(pixels, value.new_tensor(quantile)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def matte_metric_tensors(*, base_rgb: torch.Tensor, carrier_rgb: torch.Tensor,
                         alpha: torch.Tensor, coarse_target_hair_mask: torch.Tensor,
                         source_face_mask: torch.Tensor, source_skin_mask: torch.Tensor,
                         hair_support: torch.Tensor, independent_candidate: torch.Tensor,
                         hair_core: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    final = alpha * carrier_rgb + (1.0 - alpha) * base_rgb
    coarse = coarse_target_hair_mask.float().clamp(0, 1)
    support = hair_support.float().clamp(0, 1)
    boundary = (support - coarse).clamp(0, 1) + coarse * (1.0 - F.max_pool2d(-coarse, 5, stride=1, padding=2))
    binary = ((alpha < 0.05) | (alpha > 0.95)).float()
    edge = (F.max_pool2d(alpha, 3, stride=1, padding=1) - F.max_pool2d(-alpha, 3, stride=1, padding=1)).clamp(0, 1)
    visible_face = source_face_mask.float().clamp(0, 1) * (1.0 - alpha)
    visible_skin = source_skin_mask.float().clamp(0, 1) * (1.0 - alpha)
    independent = independent_candidate.float().clamp(0, 1)
    hair_core = coarse if hair_core is None else hair_core.float().clamp(0, 1)
    out = {
        "new_hair_coverage": _mean(alpha, coarse),
        "new_hair_on_face_coverage": _mean(alpha, coarse * source_face_mask.float().clamp(0, 1)),
        "visible_face_change": _mean((final - base_rgb).abs().mean(1, keepdim=True), visible_face),
        "visible_skin_change": _mean((final - base_rgb).abs().mean(1, keepdim=True), visible_skin),
        "skin_overlap_bleed": _mean((final - base_rgb).abs().mean(1, keepdim=True), source_skin_mask * alpha),
        "face_overlap_bleed": _mean((final - base_rgb).abs().mean(1, keepdim=True), source_face_mask * alpha),
        "face_boundary_alpha_jump": _mean(edge, source_face_mask * coarse),
        "alpha_edge_width_10_90": _mean(((alpha > 0.10) & (alpha < 0.90)).float(), boundary),
        "alpha_binary_fraction": _mean(binary, boundary),
        "outside_mask_alpha_fraction": _mean(alpha, support * (1.0 - coarse)),
        "independent_flyaway_candidate_count": independent.flatten(1).sum(1),
        "independent_flyaway_recall": _mean((alpha > 0.10).float(), independent),
        "hair_core_opacity": _mean(alpha, hair_core),
    }
    return {key: torch.nan_to_num(value) for key, value in out.items()}


def aggregate_matte(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"decision": "MATTE_NO_VALID_SAMPLES", "count": 0, "key_metrics": {}}
    keys = sorted(key for key in records[0] if key != "sample_id")
    med = {key: float(torch.tensor([float(row[key]) for row in records]).median()) for key in keys}
    def delta(name: str) -> float:
        return med.get(f"m1_{name}", 0.0) - med.get(f"m0_{name}", 0.0)
    confirmed = (
        delta("new_hair_coverage") >= 0.10
        and delta("new_hair_on_face_coverage") >= 0.10
        and delta("skin_overlap_bleed") <= 0.01
        and med.get("m1_face_boundary_alpha_jump", 1.0) <= med.get("m0_face_boundary_alpha_jump", 1.0) * 0.75
        and delta("independent_flyaway_recall") >= 0.05
    )
    return {"decision": "MATTE_BOTTLENECK_CONFIRMED" if confirmed else "MATTE_NOT_SUFFICIENT", "count": len(records), "key_metrics": med}


__all__ = ["aggregate_matte", "matte_metric_tensors"]
