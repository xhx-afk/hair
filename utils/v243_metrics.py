"""V2.43 confidence-core, matte, and photometric diagnostics."""

from __future__ import annotations

import torch


def _fraction(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float().clamp(0, 1), mask.float().clamp(0, 1)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def _median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != mask.shape[-2:]:
        value = value.expand(-1, -1, mask.shape[-2], mask.shape[-1])
    result = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        result.append(torch.median(pixels) if pixels.numel() else value.new_zeros(()))
    return torch.stack(result)


def _quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    if value.shape[-2:] != mask.shape[-2:]:
        value = value.expand(-1, -1, mask.shape[-2], mask.shape[-1])
    result = []
    for index in range(value.size(0)):
        pixels = value[index].flatten()[mask[index].flatten() > 0.5]
        result.append(torch.quantile(pixels, value.new_tensor(q)) if pixels.numel() else value.new_zeros(()))
    return torch.stack(result)


def v243_metric_tensors(*, target_hair_mask: torch.Tensor, hair_alpha_final: torch.Tensor,
                        safe_dense_core: torch.Tensor, uncertain_core: torch.Tensor,
                        source_skin_mask: torch.Tensor, anchor_hair_evidence: torch.Tensor,
                        strand_structure_confidence: torch.Tensor,
                        coarse_target_hair_mask: torch.Tensor | None = None,
                        hair_support: torch.Tensor | None = None,
                        flyaway_candidate: torch.Tensor | None = None,
                        final_l: torch.Tensor | None = None,
                        reference_chroma: torch.Tensor | None = None,
                        scene_illumination_ab: torch.Tensor | None = None,
                        source_hair_l: torch.Tensor | None = None,
                        reference_l: torch.Tensor | None = None,
                        **_) -> dict[str, torch.Tensor]:
    target = target_hair_mask.float().clamp(0, 1)
    alpha = hair_alpha_final.float().clamp(0, 1)
    safe, uncertain = safe_dense_core.float().clamp(0, 1), uncertain_core.float().clamp(0, 1)
    skin, evidence = source_skin_mask.float().clamp(0, 1), anchor_hair_evidence.float().clamp(0, 1)
    strand = strand_structure_confidence.float().clamp(0, 1)
    false_core = skin * (target > 0.5).float() * (evidence < 0.30).float() * (strand < 0.30).float()
    support = torch.zeros_like(target) if hair_support is None else hair_support.float().clamp(0, 1)
    candidate = torch.zeros_like(target) if flyaway_candidate is None else flyaway_candidate.float().clamp(0, 1)
    outside = support * (1.0 - (coarse_target_hair_mask if coarse_target_hair_mask is not None else target).float().clamp(0, 1))
    out = {
        "safe_dense_core_fraction": safe.flatten(1).mean(1),
        "uncertain_core_fraction": uncertain.flatten(1).mean(1),
        "safe_dense_core_alpha": _fraction(alpha, safe),
        "uncertain_core_alpha": _fraction(alpha, uncertain),
        "transition_alpha": _fraction(alpha, (target - safe).clamp(0, 1)),
        "outside_mask_flyaway_fraction": outside.flatten(1).mean(1),
        "flyaway_recovery_fraction": _fraction((alpha > 0.1).float(), candidate),
        "skin_core_false_positive_fraction": _fraction((alpha > 0.5).float(), false_core),
    }
    if final_l is not None:
        l = final_l.float()
        out["shadow_chroma_ratio"] = l.new_zeros(l.size(0))
        out["highlight_chroma_ratio"] = l.new_zeros(l.size(0))
        if reference_chroma is not None:
            q20 = _quantile(l, target, 0.20).view(-1, 1, 1, 1)
            q35 = _quantile(l, target, 0.35).view(-1, 1, 1, 1)
            q65 = _quantile(l, target, 0.65).view(-1, 1, 1, 1)
            q80 = _quantile(l, target, 0.80).view(-1, 1, 1, 1)
            shadow = target * (l <= q20).float()
            mid = target * (l >= q35).float() * (l <= q65).float()
            c = reference_chroma.float()
            out["shadow_chroma_ratio"] = _median(c, shadow) / _median(c, mid).clamp_min(1e-4)
            out["highlight_chroma_ratio"] = _median(c, target * (l >= q80).float()) / _median(c, mid).clamp_min(1e-4)
    if scene_illumination_ab is not None:
        out["scene_illumination_shift_deg"] = (torch.atan2(scene_illumination_ab[:, 1:2], scene_illumination_ab[:, 0:1]) * (180.0 / torch.pi)).flatten(1).mean(1).abs()
    if source_hair_l is not None and reference_l is not None:
        demand = (reference_l.float() - source_hair_l.float()).clamp_min(0)
        out["bleach_demand"] = demand.flatten(1).mean(1)
        out["plausibility_scale"] = 1.0 - 0.20 * (demand / 40.0).clamp(0, 1).flatten(1).mean(1)
    return out


__all__ = ["v243_metric_tensors"]
