"""V2.46 Appearance-first metrics and acceptance gates."""

from __future__ import annotations

import math
import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v240_metrics import _masked_corr, _masked_mean, _sobel


def _q(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for i in range(value.size(0)):
        pixels = value[i].flatten()[mask[i].flatten() > 0.5]
        rows.append(torch.quantile(pixels, value.new_tensor(q)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def _median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for i in range(value.size(0)):
        pixels = value[i].flatten()[mask[i].flatten() > 0.5]
        rows.append(pixels.median() if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def appearance_metric_tensors(*, carrier_rgb: torch.Tensor, output_rgb: torch.Tensor,
                              trusted_core: torch.Tensor, reference_rgb: torch.Tensor,
                              reference_hair_mask: torch.Tensor, aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    carrier_lab, output_lab, reference_lab = rgb_to_lab(carrier_rgb), rgb_to_lab(output_rgb), rgb_to_lab(reference_rgb)
    cl, ol, rl = carrier_lab[:, :1], output_lab[:, :1], reference_lab[:, :1]
    cab, oab, rab = carrier_lab[:, 1:], output_lab[:, 1:], reference_lab[:, 1:]
    mask, ref_mask = trusted_core.float().clamp(0, 1), reference_hair_mask.float().clamp(0, 1)
    out: dict[str, torch.Tensor] = {}
    for name, q in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90)):
        reference_q = _q(rl, ref_mask, q)
        out[f"reference_hair_l_{name}"] = reference_q
        out[f"l_{name}_error"] = (_q(ol, mask, q) - reference_q).abs()
    output_ab_median = torch.stack((_q(oab[:, :1], mask, .50), _q(oab[:, 1:2], mask, .50)), 1)
    reference_ab_median = torch.stack((_q(rab[:, :1], ref_mask, .50), _q(rab[:, 1:2], ref_mask, .50)), 1)
    output_c, reference_c = oab.norm(dim=1, keepdim=True), rab.norm(dim=1, keepdim=True)
    out["reference_hair_chroma"] = _q(reference_c, ref_mask, .50)
    out["median_ab_error"] = (output_ab_median - reference_ab_median).norm(dim=1)
    out["chroma_magnitude_error"] = (_q(output_c, mask, .50) - _q(reference_c, ref_mask, .50)).abs()
    output_hue, reference_hue = torch.atan2(oab[:, 1:2], oab[:, 0:1]), torch.atan2(rab[:, 1:2], rab[:, 0:1])
    hue_delta = torch.atan2(torch.sin(output_hue - _median(reference_hue, ref_mask).view(-1, 1, 1, 1)), torch.cos(output_hue - _median(reference_hue, ref_mask).view(-1, 1, 1, 1))).abs()
    out["stable_hue_error_deg"] = _median(hue_delta * (180.0 / math.pi), mask)
    delta_l = ol - cl
    delta_l_mid = gaussian_blur_v829(delta_l, 3) - gaussian_blur_v829(delta_l, 11)
    delta_l_hf = delta_l - gaussian_blur_v829(delta_l, 3)
    carrier_mid = gaussian_blur_v829(cl, 3) - gaussian_blur_v829(cl, 11)
    carrier_hf = cl - gaussian_blur_v829(cl, 3)
    delta_ab_low = aux.get("delta_ab_low", torch.zeros_like(cab))
    out.update({
        "l_residual_mean_abs": _masked_mean(delta_l.abs(), mask),
        "l_residual_p90_abs": _q(delta_l.abs(), mask, .90),
        "l_residual_mid_energy": _masked_mean(delta_l_mid.abs(), mask),
        "l_residual_hf_energy": _masked_mean(delta_l_hf.abs(), mask),
        "delta_l_to_carrier_mid_ratio": _masked_mean(delta_l_mid.abs(), mask) / _masked_mean(carrier_mid.abs(), mask).clamp_min(1e-4),
        "delta_l_to_carrier_hf_ratio": _masked_mean(delta_l_hf.abs(), mask) / _masked_mean(carrier_hf.abs(), mask).clamp_min(1e-4),
        "ab_low_residual_mean_abs": _masked_mean(delta_ab_low.norm(dim=1, keepdim=True), mask),
        "ab_low_residual_p90_abs": _q(delta_ab_low.norm(dim=1, keepdim=True), mask, .90),
        "carrier_mid_structure_corr": _masked_corr(gaussian_blur_v829(ol, 3) - gaussian_blur_v829(ol, 11), carrier_mid, mask),
        "carrier_gradient_structure_corr": _masked_corr(_sobel(ol), _sobel(cl), mask),
        "carrier_relative_mid_energy": _masked_mean((gaussian_blur_v829(ol, 3) - gaussian_blur_v829(ol, 11)).abs(), mask) / _masked_mean(carrier_mid.abs(), mask).clamp_min(1e-4),
        "carrier_relative_hf_energy": _masked_mean((ol - gaussian_blur_v829(ol, 3)).abs(), mask) / _masked_mean(carrier_hf.abs(), mask).clamp_min(1e-4),
        "trusted_core_fraction": mask.flatten(1).mean(1), "trusted_core_pixel_count": mask.flatten(1).sum(1),
        "appearance_metric_valid": (mask.flatten(1).sum(1) >= 256).float(),
        "shadow_to_midtone_chroma_ratio": _ratio(output_c, ol, mask, .20, .35, .65),
        "highlight_to_midtone_chroma_ratio": _ratio(output_c, ol, mask, .80, .35, .65),
        "shadow_valid": _region_valid(ol, mask, .20), "midtone_valid": _region_valid(ol, mask, .35, .65), "highlight_valid": _region_valid(ol, mask, .80),
    })
    if "total_gamut_scale" in aux:
        out["total_gamut_heavy_fraction"] = (aux["total_gamut_scale"] < .50).float().flatten(1).mean(1)
    return {key: torch.nan_to_num(value) for key, value in out.items()}


def _region_valid(l: torch.Tensor, mask: torch.Tensor, low: float, high: float | None = None) -> torch.Tensor:
    region = mask * (l >= _q(l, mask, low).view(-1, 1, 1, 1)).float()
    if high is not None: region = region * (l <= _q(l, mask, high).view(-1, 1, 1, 1)).float()
    return (region.flatten(1).sum(1) >= 32).float()


def _ratio(chroma: torch.Tensor, l: torch.Tensor, mask: torch.Tensor, level: float, mid_low: float, mid_high: float) -> torch.Tensor:
    region = mask * (l <= _q(l, mask, level).view(-1, 1, 1, 1)).float() if level < .5 else mask * (l >= _q(l, mask, level).view(-1, 1, 1, 1)).float()
    mid = mask * (l >= _q(l, mask, mid_low).view(-1, 1, 1, 1)).float() * (l <= _q(l, mask, mid_high).view(-1, 1, 1, 1)).float()
    return _masked_mean(chroma, region) / _masked_mean(chroma, mid).clamp_min(1e-4)


def classify_appearance(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in records if float(row.get("b1_appearance_metric_valid", 0)) > .5]
    if not valid:
        return {"decision": "V246_NO_VALID_TRUSTED_CORE", "count": 0, "key_metrics": {}}
    def med(variant: str, key: str) -> float:
        return float(torch.tensor([float(row.get(f"{variant}_{key}", 0.0)) for row in valid]).median())
    carrier_fail = med("b1", "carrier_mid_structure_corr") < .90 or med("b1", "carrier_gradient_structure_corr") < .85 or not (.90 <= med("b1", "carrier_relative_mid_energy") <= 1.15) or not (.85 <= med("b1", "carrier_relative_hf_energy") <= 1.20)
    l_fail = med("b1", "delta_l_to_carrier_mid_ratio") > .20 or med("b1", "delta_l_to_carrier_hf_ratio") > .10
    ref_improved = med("b1", "median_ab_error") <= med("b0", "median_ab_error") * .90 or med("b1", "chroma_magnitude_error") <= med("b0", "chroma_magnitude_error") * .90
    hue_ok = med("b1", "stable_hue_error_deg") <= med("b0", "stable_hue_error_deg") + 1.0
    l_ok = med("b1", "l_q50_error") <= med("b0", "l_q50_error") * 1.10
    highlight_ok = med("b1", "highlight_to_midtone_chroma_ratio") <= 1.0 and med("b1", "shadow_to_midtone_chroma_ratio") <= .90
    shading_useful = (med("b1", "highlight_to_midtone_chroma_ratio") < med("b3", "highlight_to_midtone_chroma_ratio") or med("b1", "shadow_to_midtone_chroma_ratio") < med("b3", "shadow_to_midtone_chroma_ratio")) and med("b1", "median_ab_error") <= med("b3", "median_ab_error") * 1.10
    if carrier_fail: decision = "V246_CARRIER_DAMAGE_FAIL"
    elif l_fail: decision = "V246_L_RESIDUAL_BANDLIMIT_FAIL"
    elif not (ref_improved and hue_ok and l_ok): decision = "V246_REFERENCE_FIDELITY_FAIL"
    elif not highlight_ok: decision = "V246_HIGHLIGHT_FAIL"
    else: decision = "V246_READY_FOR_VISUAL_REVIEW"
    return {"decision": decision, "count": len(valid), "carrier_preservation": "FAIL" if carrier_fail else "PASS", "l_residual_bandlimit": "FAIL" if l_fail else "PASS", "reference_fidelity": "PASS" if ref_improved and hue_ok and l_ok else "FAIL", "shading_conditioner": "USEFUL" if shading_useful else "INCONCLUSIVE", "scene_tint_enabled": False, "key_metrics": {"b1_mid_corr": med("b1", "carrier_mid_structure_corr"), "b1_gradient_corr": med("b1", "carrier_gradient_structure_corr"), "b1_l_mid_ratio": med("b1", "delta_l_to_carrier_mid_ratio"), "b1_l_hf_ratio": med("b1", "delta_l_to_carrier_hf_ratio")}}


__all__ = ["appearance_metric_tensors", "classify_appearance"]
