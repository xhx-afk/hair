"""Carrier-relative appearance metrics for the V2.45 death test."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v240_metrics import _masked_corr, _masked_mean, _sobel


def _quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        rows.append(torch.quantile(pixels, value.new_tensor(q)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def appearance_metric_tensors(*, carrier_rgb: torch.Tensor, output_rgb: torch.Tensor,
                              trusted_core: torch.Tensor, reference_l: torch.Tensor | None = None,
                              scene_illumination_ab: torch.Tensor | None = None,
                              scene_reliability: torch.Tensor | None = None,
                              shadow_scale: torch.Tensor | None = None,
                              highlight_scale: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    mask = trusted_core.float().clamp(0, 1)
    carrier_lab, output_lab = rgb_to_lab(carrier_rgb), rgb_to_lab(output_rgb)
    carrier_l, output_l = carrier_lab[:, :1], output_lab[:, :1]
    carrier_ab, output_ab = carrier_lab[:, 1:], output_lab[:, 1:]
    carrier_c, output_c = carrier_ab.norm(dim=1, keepdim=True), output_ab.norm(dim=1, keepdim=True)
    q = {name: _quantile(output_l - carrier_l, mask, level).abs() for name, level in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90))}
    mid_carrier = gaussian_blur_v829(carrier_l, 3) - gaussian_blur_v829(carrier_l, 11)
    mid_output = gaussian_blur_v829(output_l, 3) - gaussian_blur_v829(output_l, 11)
    hf_carrier = carrier_l - gaussian_blur_v829(carrier_l, 3)
    hf_output = output_l - gaussian_blur_v829(output_l, 3)
    q50_l = _quantile(carrier_l, mask, .50).view(-1, 1, 1, 1)
    shadow = mask * (carrier_l <= q50_l).float()
    highlight = mask * (carrier_l >= _quantile(carrier_l, mask, .80).view(-1, 1, 1, 1)).float()
    out = {
        **{f"l_{name}_error": value for name, value in q.items()},
        "median_ab_error": _masked_mean((output_ab - carrier_ab).norm(dim=1, keepdim=True), mask),
        "chroma_magnitude_error": _masked_mean((output_c - carrier_c).abs(), mask),
        "stable_hue_error": _masked_mean((torch.atan2(output_ab[:, 1:2], output_ab[:, 0:1]) - torch.atan2(carrier_ab[:, 1:2], carrier_ab[:, 0:1])).abs(), mask),
        "shadow_chroma_ratio": _masked_mean(output_c, shadow) / _masked_mean(carrier_c, shadow).clamp_min(1e-4),
        "highlight_chroma_ratio": _masked_mean(output_c, highlight) / _masked_mean(carrier_c, highlight).clamp_min(1e-4),
        "highlight_l_error": _masked_mean((output_l - carrier_l).abs(), highlight),
        "shadow_l_error": _masked_mean((output_l - carrier_l).abs(), shadow),
        "carrier_mid_structure_corr": _masked_corr(mid_output, mid_carrier, mask),
        "carrier_gradient_structure_corr": _masked_corr(_sobel(output_l), _sobel(carrier_l), mask),
        "carrier_relative_mid_energy": _masked_mean(mid_output.abs(), mask) / _masked_mean(mid_carrier.abs(), mask).clamp_min(1e-4),
        "carrier_relative_hf_energy": _masked_mean(hf_output.abs(), mask) / _masked_mean(hf_carrier.abs(), mask).clamp_min(1e-4),
        "total_gamut_heavy_fraction": torch.zeros(carrier_rgb.size(0), device=carrier_rgb.device),
        "scene_tint_magnitude": carrier_rgb.new_zeros(carrier_rgb.size(0)),
        "scene_tint_reliability": carrier_rgb.new_zeros(carrier_rgb.size(0)),
    }
    if reference_l is not None:
        out["reference_l_error"] = _masked_mean((output_l - reference_l).abs(), mask)
    if scene_illumination_ab is not None:
        out["scene_tint_magnitude"] = scene_illumination_ab.norm(dim=1).flatten(1).mean(1)
    if scene_reliability is not None:
        out["scene_tint_reliability"] = scene_reliability.flatten(1).mean(1)
    if shadow_scale is not None:
        out["shadow_conditioner_scale"] = _masked_mean(shadow_scale, mask)
    if highlight_scale is not None:
        out["highlight_conditioner_scale"] = _masked_mean(highlight_scale, mask)
    return {key: torch.nan_to_num(value) for key, value in out.items()}


def classify_appearance(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"decision": "APPEARANCE_NO_VALID_SAMPLES", "count": 0, "key_metrics": {}}
    def median(key: str, variant: str = "a1") -> float:
        values = [float(row.get(f"{variant}_{key}", 0.0)) for row in records]
        return float(torch.tensor(values).median())
    a1_h = median("highlight_chroma_ratio", "a1")
    a0_h = median("highlight_chroma_ratio", "a0")
    a1_quality = median("carrier_mid_structure_corr", "a1")
    a2_quality = median("carrier_mid_structure_corr", "a2")
    if a1_h > a0_h + 0.02:
        decision = "PHOTOMETRIC_MODULE_HARMFUL"
    elif a1_quality < 0.80 and median("l_q50_error", "a1") > 8.0:
        decision = "LAB_RECOLOR_LIMIT_CONFIRMED"
    elif a2_quality > a1_quality + 0.03:
        decision = "GLOBAL_SCENE_TINT_HARMFUL"
    else:
        decision = "SCENE_TINT_USEFUL" if a1_quality >= a2_quality else "CURRENT_SHADING_CONDITIONER_USEFUL"
    return {"decision": decision, "count": len(records), "key_metrics": {"a1_highlight_chroma_ratio": a1_h, "a0_highlight_chroma_ratio": a0_h, "a1_mid_structure": a1_quality, "a2_mid_structure": a2_quality}}


__all__ = ["appearance_metric_tensors", "classify_appearance"]
