"""Carrier-first Reference error estimator and continuous residual gates."""

from __future__ import annotations

import math
import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.reference_ab_statistics_v847 import _median


def _q(value: torch.Tensor, mask: torch.Tensor, level: float) -> torch.Tensor:
    value, mask = value.float(), mask.float().expand_as(value)
    rows = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        rows.append(torch.quantile(pixels, value.new_tensor(level)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(rows)


def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    x = ((value - low) / max(high - low, 1e-6)).clamp(0, 1)
    return x * x * (3.0 - 2.0 * x)


def _circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs() * (180.0 / math.pi)


class CarrierReferenceErrorEstimatorV847:
    def __call__(self, *, carrier_rgb: torch.Tensor, reference_rgb: torch.Tensor,
                 carrier_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        carrier_lab, reference_lab = rgb_to_lab(carrier_rgb), rgb_to_lab(reference_rgb)
        carrier_l, reference_l = carrier_lab[:, :1], reference_lab[:, :1]
        carrier_ab, reference_ab = carrier_lab[:, 1:], reference_lab[:, 1:]
        carrier_q = {f"carrier_l_{name}": _q(carrier_l, carrier_hair_mask, level) for name, level in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90))}
        reference_q = {f"reference_l_{name}": _q(reference_l, reference_hair_mask, level) for name, level in (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90))}
        carrier_median_ab = torch.stack((_median(carrier_ab[:, :1], carrier_hair_mask), _median(carrier_ab[:, 1:2], carrier_hair_mask)), 1)
        reference_median_ab = torch.stack((_median(reference_ab[:, :1], reference_hair_mask), _median(reference_ab[:, 1:2], reference_hair_mask)), 1)
        carrier_chroma = _median(carrier_ab.norm(dim=1, keepdim=True), carrier_hair_mask)
        reference_chroma = _median(reference_ab.norm(dim=1, keepdim=True), reference_hair_mask)
        carrier_hue = torch.atan2(carrier_median_ab[:, 1], carrier_median_ab[:, 0])
        reference_hue = torch.atan2(reference_median_ab[:, 1], reference_median_ab[:, 0])
        l_distribution_error = torch.stack([(carrier_q[f"carrier_l_{name}"] - reference_q[f"reference_l_{name}"]).abs() for name in ("q10", "q25", "q50", "q75", "q90")], 1).mean(1)
        l_error_q50 = (carrier_q["carrier_l_q50"] - reference_q["reference_l_q50"]).abs()
        ab_error = (carrier_median_ab - reference_median_ab).norm(dim=1)
        chroma_error = (carrier_chroma - reference_chroma).abs()
        hue_valid = (carrier_chroma >= 5.0) & (reference_chroma >= 5.0)
        hue_error = _circular_distance(carrier_hue, reference_hue)
        hue_error = torch.where(hue_valid, hue_error, torch.zeros_like(hue_error))
        l_gate = _smoothstep(l_distribution_error, 2.5, 8.0)
        ab_combined = .5 * ab_error + .5 * chroma_error
        ab_gate = _smoothstep(ab_combined, 2.0, 8.0)
        ab_gate = torch.where(reference_chroma >= 25.0, ab_gate * .60, ab_gate)
        ab_gate = torch.where((reference_chroma < 12.0) & (hue_valid), ab_gate.clamp(max=.70), ab_gate)
        no_op = (l_error_q50 < 2.5) & (ab_error < 2.0) & (chroma_error < 2.0) & ((~hue_valid) | (hue_error < 2.0))
        zero = torch.zeros_like(l_gate)
        return {**carrier_q, **reference_q, "carrier_median_ab": carrier_median_ab, "reference_median_ab": reference_median_ab, "carrier_chroma_median": carrier_chroma, "reference_chroma_median": reference_chroma, "carrier_hue": carrier_hue, "reference_hue": reference_hue, "l_error_q50": l_error_q50, "l_distribution_error": l_distribution_error, "ab_error": ab_error, "chroma_error": chroma_error, "hue_error_deg": hue_error, "hue_metric_valid": hue_valid.float(), "l_gate_strength": torch.where(no_op, zero, l_gate), "ab_gate_strength": torch.where(no_op, zero, ab_gate), "shading_gate_strength": zero.clone(), "plausibility_gate_strength": zero.clone(), "no_op": no_op.float()}


__all__ = ["CarrierReferenceErrorEstimatorV847"]
