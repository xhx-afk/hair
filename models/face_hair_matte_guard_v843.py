"""V2.43 skin guard: reduce uncertain matte leakage without vetoing hair."""

from __future__ import annotations

import torch


class FaceHairMatteGuardV843:
    def __init__(self, *, strength: float = 0.85) -> None:
        self.strength = float(strength)

    def __call__(self, *, alpha: torch.Tensor, source_skin_mask: torch.Tensor,
                 dense_core_confidence: torch.Tensor,
                 strand_structure_confidence: torch.Tensor,
                 anchor_hair_evidence: torch.Tensor,
                 guard_strength: float | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        skin = source_skin_mask.float().clamp(0, 1)
        dense = dense_core_confidence.float().clamp(0, 1)
        strand = strand_structure_confidence.float().clamp(0, 1)
        evidence = anchor_hair_evidence.float().clamp(0, 1)
        risk = (skin * (1.0 - dense) * (1.0 - strand) * (1.0 - evidence)).clamp(0, 1)
        strength = self.strength if guard_strength is None else float(guard_strength)
        guarded = (alpha.float().clamp(0, 1) * (1.0 - strength * risk)).clamp(0, 1)
        aux = {"guard_risk": risk, "alpha_after_guard": guarded, "alpha_final": guarded}
        return (guarded, aux) if return_aux else guarded


__all__ = ["FaceHairMatteGuardV843"]
