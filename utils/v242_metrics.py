"""V2.42 ownership and new-growth coverage metrics."""

from __future__ import annotations

import torch


def _mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value, mask = value.float(), mask.float().clamp(0, 1)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def v242_metric_tensors(*, target_hair_mask: torch.Tensor,
                        source_hair_mask: torch.Tensor,
                        source_face_mask: torch.Tensor,
                        hair_alpha_final: torch.Tensor,
                        target_hair_core: torch.Tensor | None = None,
                        target_hair_transition: torch.Tensor | None = None,
                        face_boundary_alpha_jump_map: torch.Tensor | None = None,
                        **_) -> dict[str, torch.Tensor]:
    target = target_hair_mask.float().clamp(0, 1)
    source = source_hair_mask.float().clamp(0, 1)
    new_region = target * (1.0 - source)
    core = target if target_hair_core is None else target_hair_core.float().clamp(0, 1)
    transition = (target - core).clamp(0, 1) if target_hair_transition is None else target_hair_transition.float().clamp(0, 1)
    new_core = core * (1.0 - source)
    new_transition = transition * (1.0 - source)
    alpha = hair_alpha_final.float().clamp(0, 1)
    face = source_face_mask.float().clamp(0, 1)
    core_cov = _mean(alpha, core)
    new_core_cov = _mean(alpha, new_core)
    new_cov = _mean(alpha, new_region)
    new_face_cov = _mean(alpha, new_region * face)
    out = {
        "target_core_coverage": core_cov,
        "new_hair_core_coverage": new_core_cov,
        "new_hair_coverage": new_cov,
        "new_hair_on_face_coverage": new_face_cov,
        "new_hair_region_fraction": new_region.flatten(1).mean(1),
        "new_hair_core_fraction": new_core.flatten(1).mean(1),
        "new_hair_transition_fraction": new_transition.flatten(1).mean(1),
        "ownership_loss_fraction": _mean((target - alpha).clamp_min(0), target),
        "new_hair_core_guard_loss_fraction": _mean((new_core - alpha).clamp_min(0), new_core),
        "new_hair_transition_guard_loss_fraction": _mean((new_transition - alpha).clamp_min(0), new_transition),
    }
    if face_boundary_alpha_jump_map is not None:
        out["face_boundary_alpha_jump"] = _mean(face_boundary_alpha_jump_map, target)
    else:
        out["face_boundary_alpha_jump"] = alpha.new_zeros(alpha.size(0))
    return out


__all__ = ["v242_metric_tensors"]
