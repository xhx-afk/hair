"""Separated real-parser and synthetic Z-order metrics for V2.45.1."""

from __future__ import annotations

import torch


def _masked_stats(change: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    for i in range(change.size(0)):
        pixels = change[i].flatten()[mask[i].flatten() > 0.5]
        if pixels.numel(): rows.append((pixels.mean(), torch.quantile(pixels, change.new_tensor(.90)), pixels.max(), pixels.new_tensor(float(pixels.numel()))))
        else: rows.append((change.new_zeros(()), change.new_zeros(()), change.new_zeros(()), change.new_zeros(())))
    return tuple(torch.stack(v) for v in zip(*rows))


def occluder_metric_tensors(*, base_rgb: torch.Tensor, current_rgb: torch.Tensor, protected_rgb: torch.Tensor,
                            occluder_alpha: torch.Tensor, hair_alpha: torch.Tensor,
                            hair_alpha_reference: torch.Tensor | None = None, hair_mask: torch.Tensor | None = None,
                            real_mask: torch.Tensor | None = None, synthetic_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    real = (real_mask if real_mask is not None else occluder_alpha).float().clamp(0, 1)
    synthetic = (synthetic_mask if synthetic_mask is not None else torch.zeros_like(real)).float().clamp(0, 1)
    current_change = (current_rgb - base_rgb).abs().mean(1, keepdim=True)
    protected_change = (protected_rgb - base_rgb).abs().mean(1, keepdim=True)
    cm, cp90, cmax, count = _masked_stats(current_change, real)
    pm, pp90, pmax, _ = _masked_stats(protected_change, real)
    outside = (1.0 - (occluder_alpha.float().clamp(0, 1))).clamp(0, 1)
    hair_ref = hair_alpha if hair_alpha_reference is None else hair_alpha_reference
    hair_zone = (hair_mask if hair_mask is not None else torch.ones_like(real)).float().clamp(0, 1)
    real_valid = (count >= 16).float()
    sm, scurrent_p90, smax, scount = _masked_stats(current_change, synthetic)
    smp, spp90, spmax, _ = _masked_stats(protected_change, synthetic)
    return {"current_occluder_change_from_base": cm, "protected_occluder_change_from_base": pm, "current_occluder_p90": cp90, "protected_occluder_p90": pp90, "current_occluder_max": cmax, "protected_occluder_max": pmax, "real_occluder_improvement": cm - pm, "real_occluder_pixel_count": count, "real_occluder_valid": real_valid, "synthetic_occluder_pixel_count": scount, "synthetic_current_p90": scurrent_p90, "synthetic_protected_p90": spp90, "synthetic_protected_improvement": scurrent_p90 - spp90, "hair_alpha_change_outside_occluder": ((hair_alpha - hair_ref).abs() * hair_zone * outside).flatten(1).sum(1) / (hair_zone * outside).flatten(1).sum(1).clamp_min(1e-6)}


def classify_real(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [r for r in records if bool(r.get("real_occluder_valid", False))]
    if not valid: return {"real_parser_coverage": "REAL_OCCLUDER_COVERAGE_INSUFFICIENT", "real_protection": "REAL_OCCLUDER_PROTECTION_INCONCLUSIVE", "count": 0, "key_metrics": {}}
    def med(k): return float(torch.tensor([float(r.get(k, 0)) for r in valid]).median())
    protection = "REAL_OCCLUDER_PROTECTION_CONFIRMED" if med("real_occluder_improvement") > 0.01 else "REAL_OCCLUDER_PROTECTION_INCONCLUSIVE"
    return {"real_parser_coverage": "ADEQUATE", "real_protection": protection, "count": len(valid), "key_metrics": {"current_p90": med("current_occluder_p90"), "protected_p90": med("protected_occluder_p90")}}


def classify_synthetic(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [r for r in records if float(r.get("synthetic_occluder_pixel_count", 0)) > 0]
    if not valid: return {"z_order_mechanism": "SYNTHETIC_Z_ORDER_INCONCLUSIVE", "count": 0}
    improvement = float(torch.tensor([float(r.get("synthetic_protected_improvement", 0)) for r in valid]).median())
    return {"z_order_mechanism": "OCCLUDER_Z_ORDER_MECHANISM_CONFIRMED" if improvement > 0.01 else "SYNTHETIC_Z_ORDER_INCONCLUSIVE", "count": len(valid), "key_metrics": {"protected_improvement": improvement}}


def classify_occluder(records: list[dict[str, object]]) -> dict[str, object]:
    real, synthetic = classify_real(records), classify_synthetic(records)
    return {**synthetic, **real, "z_order_mechanism": synthetic.get("z_order_mechanism"), "real_parser_coverage": real.get("real_parser_coverage"), "real_protection": real.get("real_protection"), "headphone_supported": False, "earphone_supported": False, "generic_accessory_supported": False}


__all__ = ["classify_occluder", "classify_real", "classify_synthetic", "occluder_metric_tensors"]
