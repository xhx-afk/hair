"""Correctness metrics and layered decisions for the V2.45.1 matte test."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v245_death_test_common import dilate


def _mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().expand_as(value)
    return (value.float() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def matte_metric_tensors(*, base_rgb: torch.Tensor, carrier_rgb: torch.Tensor, alpha: torch.Tensor,
                         coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                         source_skin_mask: torch.Tensor, hair_support: torch.Tensor,
                         independent_candidate: torch.Tensor, source_hair_mask: torch.Tensor | None = None,
                         hair_core: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    alpha = alpha.float().clamp(0, 1)
    target = coarse_target_hair_mask.float().clamp(0, 1)
    source_hair = torch.zeros_like(target) if source_hair_mask is None else source_hair_mask.float().clamp(0, 1)
    face, skin = source_face_mask.float().clamp(0, 1), source_skin_mask.float().clamp(0, 1)
    support = hair_support.float().clamp(0, 1)
    final = alpha * carrier_rgb + (1.0 - alpha) * base_rgb
    new_region, existing_region = target * (1.0 - source_hair), target * source_hair
    boundary_ring = (dilate(target, 2) - target).clamp(0, 1)
    target_boundary = (dilate(target, 1) - target).clamp(0, 1)
    transition, independent = ((alpha >= 0.10) & (alpha <= 0.90)).float(), independent_candidate.float().clamp(0, 1)
    candidate_count = independent.flatten(1).sum(1)
    out = {
        "target_hair_coverage": _mean(alpha, target), "existing_hair_coverage": _mean(alpha, existing_region),
        "new_hair_coverage": _mean(alpha, new_region), "new_hair_on_face_coverage": _mean(alpha, new_region * face),
        "visible_face_change": _mean((final - base_rgb).abs().mean(1, keepdim=True), face * (1 - alpha)),
        "visible_skin_change": _mean((final - base_rgb).abs().mean(1, keepdim=True), skin * (1 - alpha)),
        "skin_overlap_bleed": _mean((final - base_rgb).abs().mean(1, keepdim=True), skin * alpha),
        "face_overlap_bleed": _mean((final - base_rgb).abs().mean(1, keepdim=True), face * alpha),
        "target_boundary_alpha_jump": _mean((alpha - F.avg_pool2d(alpha, 3, 1, 1)).abs(), target_boundary),
        "face_contact_alpha_jump": _mean((alpha - F.avg_pool2d(alpha, 3, 1, 1)).abs(), target * face),
        "skin_contact_alpha_jump": _mean((alpha - F.avg_pool2d(alpha, 3, 1, 1)).abs(), target * skin),
        "boundary_intermediate_alpha_fraction": _mean(transition, boundary_ring),
        "hair_core_opacity": _mean(alpha, target if hair_core is None else hair_core.float().clamp(0, 1)),
        "outside_support_alpha": _mean(alpha, support * (1 - target)),
        "independent_flyaway_candidate_count": candidate_count, "independent_flyaway_recall": _mean((alpha > 0.10).float(), independent),
        "flyaway_metric_valid": (candidate_count >= 16).float(),
    }
    return {key: torch.nan_to_num(value) for key, value in out.items()}


def aggregate_matte(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in records if bool(row.get("m1_flyaway_metric_valid", 1))]
    if not records:
        return {"mechanism": "MATTE_NO_VALID_SAMPLES", "probe_quality": "none", "coverage_effect": False, "boundary_effect": False, "bleed_effect": "unknown", "count": 0, "key_metrics": {}}
    source = valid or records
    med = {key: float(torch.tensor([float(row[key]) for row in source if key in row]).median()) for key in source[0] if key != "sample_id" and any(key in row for row in source)}
    delta = lambda name: med.get("m1_" + name, 0.0) - med.get("m0_" + name, 0.0)
    coverage = delta("new_hair_coverage") >= 0.03 and delta("new_hair_on_face_coverage") >= 0.03
    boundary = med.get("m1_target_boundary_alpha_jump", 1.0) <= med.get("m0_target_boundary_alpha_jump", 1.0) * 0.90 or delta("boundary_intermediate_alpha_fraction") >= 0.05
    bleed = med.get("m1_skin_overlap_bleed", 0.0) <= med.get("m0_skin_overlap_bleed", 0.0) + 0.01
    mechanism, quality = (("MATTE_MECHANISM_CONFIRMED", "adequate") if coverage and boundary and bleed else ("MATTE_BOTTLENECK_CONFIRMED_BUT_PROBE_INSUFFICIENT", "coverage_only") if coverage else ("MATTE_PROBE_NO_EFFECT", "no_effect"))
    return {"mechanism": mechanism, "probe_quality": quality, "coverage_effect": coverage, "boundary_effect": boundary, "bleed_effect": "improved" if bleed and med.get("m1_skin_overlap_bleed", 0) < med.get("m0_skin_overlap_bleed", 0) else "neutral", "matte_visual_sufficiency": False, "count": len(source), "valid_flyaway_count": len(valid), "key_metrics": med}


__all__ = ["aggregate_matte", "matte_metric_tensors"]
