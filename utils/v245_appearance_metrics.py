"""Reference-fidelity, carrier-preservation, and variant decisions for V2.45.1."""

from __future__ import annotations

import math
import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v240_metrics import _masked_corr, _masked_mean, _sobel


def _quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    result = []
    for i in range(value.size(0)):
        pixels = value[i].flatten()[mask[i].flatten() > 0.5]
        result.append(torch.quantile(pixels, value.new_tensor(q)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(result)


def _safe_ratio(value: torch.Tensor, mask: torch.Tensor, min_count: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.flatten(1).sum(1)
    return _masked_mean(value, mask) , (count >= min_count).float()


def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for i in range(value.size(0)):
        pixels = value[i].flatten()[mask[i].flatten() > 0.5]
        rows.append(pixels.median() if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def appearance_metric_tensors(*, carrier_rgb: torch.Tensor, output_rgb: torch.Tensor, trusted_core: torch.Tensor,
                              reference_rgb: torch.Tensor | None = None, reference_hair_mask: torch.Tensor | None = None,
                              scene_illumination_ab: torch.Tensor | None = None, scene_reliability: torch.Tensor | None = None,
                              shadow_scale: torch.Tensor | None = None, highlight_scale: torch.Tensor | None = None,
                              total_gamut_scale: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    trusted = trusted_core.float().clamp(0, 1)
    carrier_lab, output_lab = rgb_to_lab(carrier_rgb), rgb_to_lab(output_rgb)
    cl, ol = carrier_lab[:, :1], output_lab[:, :1]
    cab, oab = carrier_lab[:, 1:], output_lab[:, 1:]
    oc, cc = oab.norm(dim=1, keepdim=True), cab.norm(dim=1, keepdim=True)
    ref_lab = rgb_to_lab(reference_rgb) if reference_rgb is not None else carrier_lab
    ref_mask = reference_hair_mask.float().clamp(0, 1) if reference_hair_mask is not None else trusted
    ref_l, ref_ab = ref_lab[:, :1], ref_lab[:, 1:]
    ref_c = ref_ab.norm(dim=1, keepdim=True)
    out: dict[str, torch.Tensor] = {}
    levels = (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90))
    for name, q in levels:
        out[f"reference_hair_l_{name}"] = _quantile(ref_l, ref_mask, q)
        out[f"l_{name}_error"] = (_quantile(ol, trusted, q) - out[f"reference_hair_l_{name}"]).abs()
    out["reference_hair_median_ab"] = torch.stack((_quantile(ref_ab[:, :1], ref_mask, .50), _quantile(ref_ab[:, 1:2], ref_mask, .50)), 1).squeeze(-1)
    out["reference_hair_chroma"] = _quantile(ref_c, ref_mask, .50)
    ref_hue = torch.atan2(ref_ab[:, 1:2], ref_ab[:, 0:1])
    out_hue = torch.atan2(oab[:, 1:2], oab[:, 0:1])
    ref_hue_median = _masked_median(ref_hue, ref_mask).view(-1, 1, 1, 1)
    hue_delta = torch.atan2(torch.sin(out_hue - ref_hue_median), torch.cos(out_hue - ref_hue_median)).abs()
    out["reference_stable_hue"] = _quantile(ref_hue, ref_mask, .50)
    out["stable_hue_error_deg"] = _masked_median(hue_delta.mul(180.0 / math.pi), trusted)
    output_median_ab = torch.stack((_quantile(oab[:, :1], trusted, .50), _quantile(oab[:, 1:2], trusted, .50)), 1).squeeze(-1)
    out["median_ab_error"] = (output_median_ab - out["reference_hair_median_ab"]).norm(dim=1)
    out["chroma_magnitude_error"] = (_quantile(oc, trusted, .50) - out["reference_hair_chroma"]).abs()
    out["reference_chroma_error"] = out["chroma_magnitude_error"]
    mid_c = _quantile(ol, trusted, .50)
    shadow = trusted * (ol <= _quantile(ol, trusted, .20).view(-1, 1, 1, 1)).float()
    midtone = trusted * (ol >= _quantile(ol, trusted, .35).view(-1, 1, 1, 1)).float() * (ol <= _quantile(ol, trusted, .65).view(-1, 1, 1, 1)).float()
    highlight = trusted * (ol >= _quantile(ol, trusted, .80).view(-1, 1, 1, 1)).float()
    shadow_c, shadow_valid = _safe_ratio(oc, shadow); mid_c_value, mid_valid = _safe_ratio(oc, midtone); high_c, high_valid = _safe_ratio(oc, highlight)
    out.update({"intra_output_shadow_to_midtone_chroma_ratio": shadow_c / mid_c_value.clamp_min(1e-4), "intra_output_highlight_to_midtone_chroma_ratio": high_c / mid_c_value.clamp_min(1e-4), "shadow_valid": shadow_valid * mid_valid, "midtone_valid": mid_valid, "highlight_valid": high_valid * mid_valid})
    mid_carrier = gaussian_blur_v829(cl, 3) - gaussian_blur_v829(cl, 11); mid_output = gaussian_blur_v829(ol, 3) - gaussian_blur_v829(ol, 11)
    hf_carrier, hf_output = cl - gaussian_blur_v829(cl, 3), ol - gaussian_blur_v829(ol, 3)
    out.update({"carrier_mid_structure_corr": _masked_corr(mid_output, mid_carrier, trusted), "carrier_gradient_structure_corr": _masked_corr(_sobel(ol), _sobel(cl), trusted), "carrier_relative_mid_energy": _masked_mean(mid_output.abs(), trusted) / _masked_mean(mid_carrier.abs(), trusted).clamp_min(1e-4), "carrier_relative_hf_energy": _masked_mean(hf_output.abs(), trusted) / _masked_mean(hf_carrier.abs(), trusted).clamp_min(1e-4), "trusted_core_fraction": trusted.flatten(1).mean(1), "trusted_core_pixel_count": trusted.flatten(1).sum(1), "appearance_metric_valid": (trusted.flatten(1).sum(1) >= 256).float()})
    if total_gamut_scale is not None:
        out["total_gamut_heavy_fraction"] = (total_gamut_scale.float() < 0.50).float().flatten(1).mean(1)
    if scene_illumination_ab is not None:
        out["scene_tint_magnitude"] = scene_illumination_ab.norm(dim=1).flatten(1).mean(1)
    if scene_reliability is not None:
        out["scene_tint_reliability"] = scene_reliability.float().flatten(1).mean(1)
    if shadow_scale is not None: out["shadow_conditioner_scale"] = _masked_mean(shadow_scale, trusted)
    if highlight_scale is not None: out["highlight_conditioner_scale"] = _masked_mean(highlight_scale, trusted)
    return {k: torch.nan_to_num(v) for k, v in out.items()}


def classify_appearance(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [r for r in records if bool(r.get("a1_appearance_metric_valid", 1))]
    if not valid: return {"photometric_overall": "APPEARANCE_INVALID_NO_TRUSTED_CORE", "scene_tint": "GLOBAL_SCENE_TINT_INCONCLUSIVE", "shading_conditioner": "SHADING_CONDITIONER_INCONCLUSIVE", "carrier_preservation": "UNKNOWN", "reference_fidelity": "UNKNOWN", "count": 0, "key_metrics": {}}
    def med(v: str, variant: str = "a1") -> float: return float(torch.tensor([float(r.get(f"{variant}_{v}", 0.0)) for r in valid]).median())
    a1_err, a0_err = med("median_ab_error", "a1"), med("median_ab_error", "a0")
    a1_h, a2_h = med("stable_hue_error_deg", "a1"), med("stable_hue_error_deg", "a2")
    a1_c, a2_c = med("reference_chroma_error", "a1"), med("reference_chroma_error", "a2")
    carrier_pass = med("carrier_mid_structure_corr", "a1") >= 0.80 and med("carrier_gradient_structure_corr", "a1") >= 0.75
    scene = "GLOBAL_SCENE_TINT_HARMFUL" if (a2_h <= a1_h - 2.0 or a2_c <= a1_c * 0.90) and med("carrier_mid_structure_corr", "a2") >= med("carrier_mid_structure_corr", "a1") - 0.03 else "GLOBAL_SCENE_TINT_USEFUL" if (a1_h < a2_h - 2.0 or a1_c < a2_c * 0.90) else "GLOBAL_SCENE_TINT_INCONCLUSIVE"
    shading = "SHADING_CONDITIONER_USEFUL" if med("intra_output_highlight_to_midtone_chroma_ratio", "a1") <= med("intra_output_highlight_to_midtone_chroma_ratio", "a3") and med("intra_output_shadow_to_midtone_chroma_ratio", "a1") <= med("intra_output_shadow_to_midtone_chroma_ratio", "a3") and a1_err <= med("median_ab_error", "a3") * 1.10 else "SHADING_CONDITIONER_HARMFUL" if a1_err > med("median_ab_error", "a3") * 1.10 else "SHADING_CONDITIONER_INCONCLUSIVE"
    overall = "PHOTOMETRIC_INVALID_DUE_TO_CARRIER_DAMAGE" if not carrier_pass else "PHOTOMETRIC_HAS_VALUE" if a1_err < a0_err * 0.90 else "PHOTOMETRIC_NOT_HELPFUL"
    return {"photometric_overall": overall, "scene_tint": scene, "shading_conditioner": shading, "carrier_preservation": "PASS" if carrier_pass else "FAIL", "reference_fidelity": "IMPROVED" if a1_err < a0_err else "NOT_IMPROVED", "count": len(valid), "key_metrics": {"a1_reference_error": a1_err, "a0_reference_error": a0_err, "a1_hue_error": a1_h, "a2_hue_error": a2_h}}


__all__ = ["appearance_metric_tensors", "classify_appearance"]
