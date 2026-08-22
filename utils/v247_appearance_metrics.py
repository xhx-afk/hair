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


def _median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > .5]
        rows.append(pixels.median() if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def _ratio(chroma: torch.Tensor, luma: torch.Tensor, mask: torch.Tensor, low: float, high: float | None = None) -> torch.Tensor:
    qlow = _q(luma, mask, low).view(-1, 1, 1, 1)
    region = mask * ((luma <= qlow) if high is None else ((luma >= qlow) & (luma <= _q(luma, mask, high).view(-1, 1, 1, 1)))).float()
    return _masked_mean(chroma, region)


def appearance_metric_tensors(*, carrier_rgb: torch.Tensor, output_rgb: torch.Tensor,
                              trusted_core: torch.Tensor, reference_rgb: torch.Tensor,
                              reference_hair_mask: torch.Tensor, aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    carrier_lab, output_lab, reference_lab = rgb_to_lab(carrier_rgb), rgb_to_lab(output_rgb), rgb_to_lab(reference_rgb)
    cl, ol, rl = carrier_lab[:, :1], output_lab[:, :1], reference_lab[:, :1]
    cab, oab, rab = carrier_lab[:, 1:], output_lab[:, 1:], reference_lab[:, 1:]
    mask, ref_mask = trusted_core.float().clamp(0, 1), reference_hair_mask.float().clamp(0, 1)
    out: dict[str, torch.Tensor] = {}
    for name, level in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90)):
        carrier_q, reference_q = _q(cl, mask, level), _q(rl, ref_mask, level)
        out[f"carrier_l_{name}"] = carrier_q; out[f"reference_l_{name}"] = reference_q; out[f"l_{name}_error"] = (carrier_q - reference_q).abs()
    carrier_ab_median = torch.stack((_q(cab[:, :1], mask, .50), _q(cab[:, 1:2], mask, .50)), 1)
    reference_ab_median = torch.stack((_q(rab[:, :1], ref_mask, .50), _q(rab[:, 1:2], ref_mask, .50)), 1)
    carrier_c, output_c, reference_c = cab.norm(dim=1, keepdim=True), oab.norm(dim=1, keepdim=True), rab.norm(dim=1, keepdim=True)
    out["carrier_median_ab"] = carrier_ab_median; out["reference_median_ab"] = reference_ab_median; out["median_ab_error"] = (torch.stack((_q(oab[:, :1], mask, .50), _q(oab[:, 1:2], mask, .50)), 1) - reference_ab_median).norm(dim=1)
    out["reference_chroma_median"] = _q(reference_c, ref_mask, .50); out["chroma_error"] = (_q(output_c, mask, .50) - out["reference_chroma_median"]).abs(); out["carrier_chroma_error"] = (_q(carrier_c, mask, .50) - out["reference_chroma_median"]).abs()
    carrier_hue, reference_hue = torch.atan2(carrier_ab_median[:, 1], carrier_ab_median[:, 0]), torch.atan2(reference_ab_median[:, 1], reference_ab_median[:, 0])
    output_hue = torch.atan2(_q(oab[:, 1:2], mask, .50), _q(oab[:, :1], mask, .50))
    out["stable_hue_error_deg"] = torch.atan2(torch.sin(output_hue - reference_hue), torch.cos(output_hue - reference_hue)).abs() * (180.0 / math.pi)
    delta_l = ol - cl; carrier_mid = gaussian_blur_v829(cl, 3) - gaussian_blur_v829(cl, 11); output_mid = gaussian_blur_v829(ol, 3) - gaussian_blur_v829(ol, 11); delta_mid = gaussian_blur_v829(delta_l, 3) - gaussian_blur_v829(delta_l, 11); delta_hf = delta_l - gaussian_blur_v829(delta_l, 3)
    out.update({"carrier_mid_structure_corr": _masked_corr(output_mid, carrier_mid, mask), "carrier_gradient_structure_corr": _masked_corr(_sobel(ol), _sobel(cl), mask), "carrier_relative_mid_energy": _masked_mean(output_mid.abs(), mask) / _masked_mean(carrier_mid.abs(), mask).clamp_min(1e-4), "carrier_relative_hf_energy": _masked_mean((ol - gaussian_blur_v829(ol, 3)).abs(), mask) / _masked_mean((cl - gaussian_blur_v829(cl, 3)).abs(), mask).clamp_min(1e-4), "l_residual_mean_abs": _masked_mean(delta_l.abs(), mask), "l_residual_mid_energy": _masked_mean(delta_mid.abs(), mask), "l_residual_hf_energy": _masked_mean(delta_hf.abs(), mask), "delta_l_to_carrier_mid_ratio": _masked_mean(delta_mid.abs(), mask) / _masked_mean(carrier_mid.abs(), mask).clamp_min(1e-4), "delta_l_to_carrier_hf_ratio": _masked_mean(delta_hf.abs(), mask) / _masked_mean((cl - gaussian_blur_v829(cl, 3)).abs(), mask).clamp_min(1e-4), "trusted_core_pixel_count": mask.flatten(1).sum(1), "trusted_core_fraction": mask.flatten(1).mean(1), "appearance_metric_valid": (mask.flatten(1).sum(1) >= 256).float(), "l_gate_strength": aux.get("l_gate_strength", torch.zeros_like(mask)).flatten(1).mean(1), "ab_gate_strength": aux.get("ab_gate_strength", torch.zeros_like(mask)).flatten(1).mean(1), "shading_gate_strength": aux.get("shading_gate_strength", torch.zeros_like(mask)).flatten(1).mean(1), "plausibility_gate_strength": aux.get("plausibility_gate_strength", torch.zeros_like(mask)).flatten(1).mean(1), "l_delta_clamp_fraction": aux.get("l_delta_clamp_fraction", torch.zeros(mask.size(0), device=mask.device)), "l_delta_p90": aux.get("l_delta_p90", torch.zeros(mask.size(0), device=mask.device)), "l_delta_mean_abs": aux.get("l_delta_mean_abs", torch.zeros(mask.size(0), device=mask.device)), "ab_correction_magnitude": aux.get("ab_correction_magnitude", torch.zeros(mask.size(0), device=mask.device)), "non_hair_rgb_change": _masked_mean((output_rgb - carrier_rgb).abs().mean(1, keepdim=True), 1.0 - aux.get("target_hair_soft", mask))})
    out["shadow_to_midtone_chroma_ratio"] = _ratio(output_c, ol, mask, .20) / _ratio(output_c, ol, mask, .35, .65).clamp_min(1e-4); out["highlight_to_midtone_chroma_ratio"] = _ratio(output_c, ol, mask, .80, None) / _ratio(output_c, ol, mask, .35, .65).clamp_min(1e-4)
    if "total_gamut_scale" in aux: out["total_gamut_heavy_fraction"] = (aux["total_gamut_scale"] < .50).float().flatten(1).mean(1)
    out["noop"] = ((out["l_gate_strength"] < .05) & (out["ab_gate_strength"] < .05) & (out["shading_gate_strength"] < .05) & (out["plausibility_gate_strength"] < .05)).float()
    return {key: torch.nan_to_num(value) for key, value in out.items()}


def _med(records: list[dict[str, object]], key: str, variant: str) -> float:
    return float(torch.tensor([float(row.get(f"{variant}_{key}", 0.0)) for row in records]).median()) if records else 0.0


def _useful(records: list[dict[str, object]], variant: str) -> bool:
    return _med(records, "median_ab_error", variant) <= _med(records, "median_ab_error", "c0") * .90 or _med(records, "chroma_error", variant) <= _med(records, "chroma_error", "c0") * .90


def classify_components(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in records if float(row.get("c0_appearance_metric_valid", 0)) > .5]
    if not valid: return {"decision": "V247_CARRIER_CONTRACT_FAIL", "count": 0}
    carrier_fail = _med(valid, "carrier_mid_structure_corr", "c5") < .95 or _med(valid, "carrier_gradient_structure_corr", "c5") < .95 or not (.95 <= _med(valid, "carrier_relative_mid_energy", "c5") <= 1.10) or not (.90 <= _med(valid, "carrier_relative_hf_energy", "c5") <= 1.15) or _med(valid, "non_hair_rgb_change", "c5") > 1e-5
    l_useful = _med(valid, "l_q50_error", "c1") <= _med(valid, "l_q50_error", "c0") * .90 and _med(valid, "median_ab_error", "c1") <= _med(valid, "median_ab_error", "c0") * 1.05
    ab_useful = _useful(valid, "c2") and _med(valid, "stable_hue_error_deg", "c2") <= _med(valid, "stable_hue_error_deg", "c0") + 1.0
    interaction = "L_AB_INTERACTION_HARMFUL" if _med(valid, "median_ab_error", "c3") > min(_med(valid, "median_ab_error", "c1"), _med(valid, "median_ab_error", "c2")) * 1.05 and _med(valid, "chroma_error", "c3") > min(_med(valid, "chroma_error", "c1"), _med(valid, "chroma_error", "c2")) * 1.05 else "L_AB_INTERACTION_OK"
    shading = "SHADING_MODULE_USEFUL" if (_med(valid, "shadow_to_midtone_chroma_ratio", "c4") < _med(valid, "shadow_to_midtone_chroma_ratio", "c3") or _med(valid, "highlight_to_midtone_chroma_ratio", "c4") < _med(valid, "highlight_to_midtone_chroma_ratio", "c3")) and _med(valid, "median_ab_error", "c4") <= _med(valid, "median_ab_error", "c3") * 1.05 else "SHADING_MODULE_HARMFUL"
    plausibility = "PLAUSIBILITY_MODULE_HARMFUL" if _med(valid, "chroma_error", "c5") > _med(valid, "chroma_error", "c4") * 1.05 else "PLAUSIBILITY_MODULE_INCONCLUSIVE" if _med(valid, "plausibility_gate_strength", "c5") < .05 else "PLAUSIBILITY_MODULE_USEFUL"
    noop = sum(float(row.get("c5_noop", 0.0)) for row in valid) / len(valid)
    if carrier_fail: decision = "V247_CARRIER_CONTRACT_FAIL"
    elif not (l_useful or ab_useful): decision = "V247_NO_USEFUL_COMPONENT_FOUND"
    elif l_useful and ab_useful and shading == "SHADING_MODULE_USEFUL" and plausibility != "PLAUSIBILITY_MODULE_HARMFUL": decision = "V247_READY_FOR_FORMAL_INTEGRATION"
    elif l_useful and ab_useful: decision = "V247_ERROR_AWARE_L_AB"
    elif l_useful: decision = "V247_ERROR_AWARE_L_ONLY"
    else: decision = "V247_ERROR_AWARE_AB_ONLY"
    return {"decision": decision, "count": len(valid), "carrier": {"decision": "V247_CARRIER_CONTRACT_FAIL" if carrier_fail else "PASS"}, "l_module": {"decision": "L_MODULE_USEFUL" if l_useful else "L_MODULE_NOT_HELPFUL"}, "ab_module": {"decision": "AB_MODULE_USEFUL" if ab_useful else "AB_MODULE_NOT_HELPFUL"}, "l_ab_interaction": {"decision": interaction}, "shading_module": {"decision": shading}, "plausibility_module": {"decision": plausibility}, "noop_sample_fraction": noop, "recommended_active_components": [name for name, enabled in (("L", l_useful), ("AB", ab_useful), ("Shading", shading == "SHADING_MODULE_USEFUL"), ("Plausibility", plausibility == "PLAUSIBILITY_MODULE_USEFUL")) if enabled]}


__all__ = ["appearance_metric_tensors", "classify_components"]
