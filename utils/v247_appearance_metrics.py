"""Carrier-first component attribution metrics and gates for V2.47."""

from __future__ import annotations

import math
import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v240_metrics import _masked_corr, _masked_mean, _sobel


def _q(value: torch.Tensor, mask: torch.Tensor, level: float) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > .5]
        rows.append(torch.quantile(pixels, value.new_tensor(level)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def _batch_scalar(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize scalar-like aux fields to one value per batch element."""
    if value.ndim == 0:
        return value.reshape(1).expand(batch_size)
    if value.ndim == 1:
        return value
    return value.reshape(value.size(0), -1).mean(1)


def _median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > .5]
        rows.append(pixels.median() if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def _lower_tail_mean(value, luma, mask, quantile):
    q = _q(luma, mask, quantile).view(-1, 1, 1, 1)
    return _masked_mean(value, mask.float() * (luma <= q).float())


def _upper_tail_mean(value, luma, mask, quantile):
    q = _q(luma, mask, quantile).view(-1, 1, 1, 1)
    return _masked_mean(value, mask.float() * (luma >= q).float())


def _middle_band_mean(value, luma, mask, low_q, high_q):
    q_low = _q(luma, mask, low_q).view(-1, 1, 1, 1)
    q_high = _q(luma, mask, high_q).view(-1, 1, 1, 1)
    region = mask.float() * (luma >= q_low).float() * (luma <= q_high).float()
    return _masked_mean(value, region)


def appearance_metric_tensors(*, carrier_rgb: torch.Tensor, output_rgb: torch.Tensor,
                              trusted_core: torch.Tensor, reference_rgb: torch.Tensor,
                              reference_hair_mask: torch.Tensor, aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    carrier_lab, output_lab, reference_lab = rgb_to_lab(carrier_rgb), rgb_to_lab(output_rgb), rgb_to_lab(reference_rgb)
    cl, ol, rl = carrier_lab[:, :1], output_lab[:, :1], reference_lab[:, :1]
    cab, oab, rab = carrier_lab[:, 1:], output_lab[:, 1:], reference_lab[:, 1:]
    mask, ref_mask = trusted_core.float().clamp(0, 1), reference_hair_mask.float().clamp(0, 1)
    out: dict[str, torch.Tensor] = {}
    for name, level in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90)):
        carrier_q, output_q, reference_q = _q(cl, mask, level), _q(ol, mask, level), _q(rl, ref_mask, level)
        out[f"carrier_l_{name}"] = carrier_q; out[f"output_l_{name}"] = output_q; out[f"reference_l_{name}"] = reference_q; out[f"l_{name}_error"] = (output_q - reference_q).abs(); out[f"carrier_l_{name}_error"] = (carrier_q - reference_q).abs()
    carrier_ab_median = torch.stack((_q(cab[:, :1], mask, .50), _q(cab[:, 1:2], mask, .50)), 1)
    reference_ab_median = torch.stack((_q(rab[:, :1], ref_mask, .50), _q(rab[:, 1:2], ref_mask, .50)), 1)
    carrier_c, output_c, reference_c = cab.norm(dim=1, keepdim=True), oab.norm(dim=1, keepdim=True), rab.norm(dim=1, keepdim=True)
    out["carrier_median_ab"] = carrier_ab_median; out["reference_median_ab"] = reference_ab_median; out["median_ab_error"] = (torch.stack((_q(oab[:, :1], mask, .50), _q(oab[:, 1:2], mask, .50)), 1) - reference_ab_median).norm(dim=1)
    out["reference_chroma_median"] = _q(reference_c, ref_mask, .50); out["chroma_error"] = (_q(output_c, mask, .50) - out["reference_chroma_median"]).abs(); out["carrier_chroma_error"] = (_q(carrier_c, mask, .50) - out["reference_chroma_median"]).abs()
    carrier_hue, reference_hue = torch.atan2(carrier_ab_median[:, 1], carrier_ab_median[:, 0]), torch.atan2(reference_ab_median[:, 1], reference_ab_median[:, 0])
    output_hue = torch.atan2(_q(oab[:, 1:2], mask, .50), _q(oab[:, :1], mask, .50))
    out["stable_hue_error_deg"] = torch.atan2(torch.sin(output_hue - reference_hue), torch.cos(output_hue - reference_hue)).abs() * (180.0 / math.pi)
    out["carrier_stable_hue_error_deg"] = torch.atan2(torch.sin(carrier_hue - reference_hue), torch.cos(carrier_hue - reference_hue)).abs() * (180.0 / math.pi)
    delta_l = ol - cl; carrier_mid = gaussian_blur_v829(cl, 3) - gaussian_blur_v829(cl, 11); output_mid = gaussian_blur_v829(ol, 3) - gaussian_blur_v829(ol, 11); delta_mid = gaussian_blur_v829(delta_l, 3) - gaussian_blur_v829(delta_l, 11); delta_hf = delta_l - gaussian_blur_v829(delta_l, 3)
    target_soft = aux.get("target_hair_soft", mask)
    strict_non_hair = target_soft < .01
    boundary = (target_soft >= .01) & (target_soft < .5)
    change = (output_rgb - carrier_rgb).abs().mean(1, keepdim=True)
    batch_size = mask.size(0)
    scalar_keys = ("l_gate_strength", "ab_gate_strength", "shadow_gate_strength", "highlight_gate_strength", "shading_gate_strength", "plausibility_gate_strength", "l_delta_clamp_fraction", "l_delta_p90", "l_delta_mean_abs", "ab_correction_magnitude")
    for key in scalar_keys:
        default = torch.zeros(batch_size, device=mask.device)
        out[key] = _batch_scalar(aux.get(key, default), batch_size)
    out.update({"carrier_mid_structure_corr": _masked_corr(output_mid, carrier_mid, mask), "carrier_gradient_structure_corr": _masked_corr(_sobel(ol), _sobel(cl), mask), "carrier_relative_mid_energy": _masked_mean(output_mid.abs(), mask) / _masked_mean(carrier_mid.abs(), mask).clamp_min(1e-4), "carrier_relative_hf_energy": _masked_mean((ol - gaussian_blur_v829(ol, 3)).abs(), mask) / _masked_mean((cl - gaussian_blur_v829(cl, 3)).abs(), mask).clamp_min(1e-4), "l_residual_mean_abs": _masked_mean(delta_l.abs(), mask), "l_residual_mid_energy": _masked_mean(delta_mid.abs(), mask), "l_residual_hf_energy": _masked_mean(delta_hf.abs(), mask), "delta_l_to_carrier_mid_ratio": _masked_mean(delta_mid.abs(), mask) / _masked_mean(carrier_mid.abs(), mask).clamp_min(1e-4), "delta_l_to_carrier_hf_ratio": _masked_mean(delta_hf.abs(), mask) / _masked_mean((cl - gaussian_blur_v829(cl, 3)).abs(), mask).clamp_min(1e-4), "trusted_core_pixel_count": mask.flatten(1).gt(.5).sum(1), "trusted_core_fraction": mask.flatten(1).mean(1), "appearance_metric_valid": (mask.flatten(1).gt(.5).sum(1) >= 256).float(), "gate_metric_valid": _batch_scalar(aux.get("gate_metric_valid", torch.zeros(batch_size, device=mask.device)), batch_size), "stats_mask_pixel_count": _batch_scalar(aux.get("stats_mask_pixel_count", torch.zeros(batch_size, device=mask.device)), batch_size), "non_hair_rgb_change": _masked_mean(change, 1.0 - target_soft), "strict_non_hair_max_rgb_change": (change * strict_non_hair).flatten(1).amax(1), "strict_non_hair_mean_rgb_change": _masked_mean(change, strict_non_hair), "boundary_transition_rgb_change": _masked_mean(change, boundary)})
    shadow_region = mask * (ol <= _q(ol, mask, .20).view(-1,1,1,1)).float()
    mid_region = mask * (ol >= _q(ol, mask, .35).view(-1,1,1,1)).float() * (ol <= _q(ol, mask, .65).view(-1,1,1,1)).float()
    highlight_region = mask * (ol >= _q(ol, mask, .80).view(-1,1,1,1)).float()
    out["shadow_pixel_count"] = shadow_region.flatten(1).gt(.5).sum(1)
    out["midtone_pixel_count"] = mid_region.flatten(1).gt(.5).sum(1)
    out["highlight_pixel_count"] = highlight_region.flatten(1).gt(.5).sum(1)
    region_valid = (out["shadow_pixel_count"] >= 32) & (out["midtone_pixel_count"] >= 32) & (out["highlight_pixel_count"] >= 32)
    out["shadow_metric_valid"] = (out["shadow_pixel_count"] >= 32).float(); out["midtone_metric_valid"] = (out["midtone_pixel_count"] >= 32).float(); out["highlight_metric_valid"] = (out["highlight_pixel_count"] >= 32).float(); out["shading_metric_valid"] = region_valid.float()
    shadow_c = _lower_tail_mean(output_c, ol, mask, .20); mid_c = _middle_band_mean(output_c, ol, mask, .35, .65); highlight_c = _upper_tail_mean(output_c, ol, mask, .80)
    out["shadow_to_midtone_chroma_ratio"] = torch.where(out["shading_metric_valid"] > .5, shadow_c / mid_c.clamp_min(1e-4), torch.full_like(shadow_c, float("nan")))
    out["highlight_to_midtone_chroma_ratio"] = torch.where(out["shading_metric_valid"] > .5, highlight_c / mid_c.clamp_min(1e-4), torch.full_like(highlight_c, float("nan")))
    if "total_gamut_scale" in aux:
        out["total_gamut_heavy_fraction"] = (aux["total_gamut_scale"] < .50).float().flatten(1).mean(1)
        out["trusted_core_gamut_heavy_fraction"] = _masked_mean((aux["total_gamut_scale"] < .50).float(), mask)
        out["target_hair_gamut_heavy_fraction"] = _masked_mean((aux["total_gamut_scale"] < .50).float(), (target_soft >= .01).float())
    out["noop"] = ((out["l_gate_strength"] < .05) & (out["ab_gate_strength"] < .05) & (out["shading_gate_strength"] < .05) & (out["plausibility_gate_strength"] < .05)).float()
    return out


def _med(records: list[dict[str, object]], key: str, variant: str) -> float:
    values = [float(row.get(f"{variant}_{key}", float("nan"))) for row in records]
    values = [value for value in values if math.isfinite(value)]
    return float(torch.tensor(values).median()) if values else float("nan")


def _useful(records: list[dict[str, object]], variant: str) -> bool:
    return (_med(records, "median_ab_error", variant) <= _med(records, "median_ab_error", "c0") * .90 or _med(records, "chroma_error", variant) <= _med(records, "chroma_error", "c0") * .90)


def _l_distribution(records: list[dict[str, object]], variant: str) -> float:
    return sum(_med(records, f"l_{quantile}_error", variant) for quantile in ("q10", "q25", "q50", "q75", "q90")) / 5.0


def _improvement(before: float, after: float) -> float:
    return 1.0 - after / max(abs(before), 1e-6)


def _distance_to_range(value: float, low: float, high: float) -> float:
    return max(low - value, value - high, 0.0)


def classify_components(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in records if all(float(row.get(f"{variant}_appearance_metric_valid", 0)) > .5 for variant in ("c0", "c1", "c2", "c3", "c4", "c5"))]
    if not valid: return {"decision": "V247_NO_VALID_SAMPLES", "count": 0}
    c0_l_q50, c1_l_q50 = _med(valid, "l_q50_error", "c0"), _med(valid, "l_q50_error", "c1")
    c0_l_dist, c1_l_dist = _l_distribution(valid, "c0"), _l_distribution(valid, "c1")
    l_improves = (c1_l_q50 <= c0_l_q50 * .90) or (c1_l_dist <= c0_l_dist * .90)
    l_preserves_ab = _med(valid, "median_ab_error", "c1") <= _med(valid, "median_ab_error", "c0") * 1.05
    l_preserves_chroma = _med(valid, "chroma_error", "c1") <= _med(valid, "chroma_error", "c0") * 1.05
    l_conflict = _med(valid, "l_delta_clamp_fraction", "c1") > .20
    l_useful = l_improves and l_preserves_ab and l_preserves_chroma and not l_conflict
    ab_useful = _useful(valid, "c2") and _med(valid, "stable_hue_error_deg", "c2") <= _med(valid, "stable_hue_error_deg", "c0") + 1.0
    interaction = "L_AB_INTERACTION_HARMFUL" if _med(valid, "median_ab_error", "c3") > min(_med(valid, "median_ab_error", "c1"), _med(valid, "median_ab_error", "c2")) * 1.05 and _med(valid, "chroma_error", "c3") > min(_med(valid, "chroma_error", "c1"), _med(valid, "chroma_error", "c2")) * 1.05 else "L_AB_INTERACTION_OK"
    shading_before = _distance_to_range(_med(valid, "shadow_to_midtone_chroma_ratio", "c3"), .70, .95) + _distance_to_range(_med(valid, "highlight_to_midtone_chroma_ratio", "c3"), .80, 1.05)
    shading_after = _distance_to_range(_med(valid, "shadow_to_midtone_chroma_ratio", "c4"), .70, .95) + _distance_to_range(_med(valid, "highlight_to_midtone_chroma_ratio", "c4"), .80, 1.05)
    shading_gain = _improvement(shading_before, shading_after)
    shading_fidelity = _med(valid, "median_ab_error", "c4") <= _med(valid, "median_ab_error", "c3") * 1.05 and _med(valid, "chroma_error", "c4") <= _med(valid, "chroma_error", "c3") * 1.05 and _med(valid, "stable_hue_error_deg", "c4") <= _med(valid, "stable_hue_error_deg", "c3") + 1.0
    shading = "SHADING_MODULE_USEFUL" if shading_gain >= .10 and shading_fidelity else ("SHADING_MODULE_HARMFUL" if not shading_fidelity else "SHADING_MODULE_NOT_HELPFUL")
    high = [row for row in valid if row.get("reference_chroma_group") == "high_chroma"] or valid
    plaus_gain_chroma = _improvement(_med(high, "chroma_error", "c4"), _med(high, "chroma_error", "c5"))
    plaus_gain_highlight = _improvement(_distance_to_range(_med(high, "highlight_to_midtone_chroma_ratio", "c4"), .80, 1.05), _distance_to_range(_med(high, "highlight_to_midtone_chroma_ratio", "c5"), .80, 1.05))
    plaus_fidelity = _med(high, "median_ab_error", "c5") <= _med(high, "median_ab_error", "c4") * 1.05 and _med(high, "stable_hue_error_deg", "c5") <= _med(high, "stable_hue_error_deg", "c4") + 1.0
    plausibility = "PLAUSIBILITY_MODULE_HARMFUL" if _med(high, "chroma_error", "c5") > _med(high, "chroma_error", "c4") * 1.05 or not plaus_fidelity else ("PLAUSIBILITY_MODULE_USEFUL" if (plaus_gain_chroma >= .05 or plaus_gain_highlight >= .10) else "PLAUSIBILITY_MODULE_NOT_HELPFUL")
    noop = sum(float(row.get("c5_noop", 0.0)) for row in valid) / len(valid)
    if not (l_useful or ab_useful): decision = "V247_CARRIER_ONLY"
    elif l_useful and ab_useful and shading == "SHADING_MODULE_USEFUL": decision = "V247_ERROR_AWARE_L_AB_SHADING"
    elif l_useful and ab_useful: decision = "V247_ERROR_AWARE_L_AB"
    elif l_useful: decision = "V247_ERROR_AWARE_L_ONLY"
    else: decision = "V247_ERROR_AWARE_AB_ONLY"
    l_q50_improvement = _improvement(c0_l_q50, c1_l_q50)
    l_distribution_improvement = _improvement(c0_l_dist, c1_l_dist)
    return {"decision": decision, "count": len(valid), "validity": {variant: sum(float(row.get(f"{variant}_appearance_metric_valid", 0)) > .5 for row in records) for variant in ("c0", "c1", "c2", "c3", "c4", "c5")}, "l_module": {"decision": "L_MODULE_USEFUL" if l_useful else ("L_TARGET_CONFLICT" if l_conflict else "L_MODULE_NOT_HELPFUL"), "q50_improvement": l_q50_improvement, "distribution_improvement": l_distribution_improvement, "ab_degradation": _med(valid, "median_ab_error", "c1") / max(_med(valid, "median_ab_error", "c0"), 1e-6) - 1.0, "chroma_degradation": _med(valid, "chroma_error", "c1") / max(_med(valid, "chroma_error", "c0"), 1e-6) - 1.0, "clamp_fraction": _med(valid, "l_delta_clamp_fraction", "c1")}, "ab_module": {"decision": "AB_MODULE_USEFUL" if ab_useful else "AB_MODULE_NOT_HELPFUL", "ab_improvement": _improvement(_med(valid, "median_ab_error", "c0"), _med(valid, "median_ab_error", "c2")), "chroma_improvement": _improvement(_med(valid, "chroma_error", "c0"), _med(valid, "chroma_error", "c2"))}, "l_ab_interaction": {"decision": interaction}, "shading_module": {"decision": shading, "distance_before": shading_before, "distance_after": shading_after, "improvement": shading_gain}, "plausibility_module": {"decision": plausibility, "high_chroma_chroma_improvement": plaus_gain_chroma, "high_chroma_highlight_improvement": plaus_gain_highlight}, "noop_sample_fraction": noop, "recommended_active_components": [name for name, enabled in (("L", l_useful), ("AB", ab_useful), ("Shading", shading == "SHADING_MODULE_USEFUL"), ("Plausibility", plausibility == "PLAUSIBILITY_MODULE_USEFUL")) if enabled]}


__all__ = ["appearance_metric_tensors", "classify_components"]
