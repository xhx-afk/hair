"""V2.42 topology-first matte refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TargetHairMatteRefinerV842:
    def __init__(self, *, core_floor: float = 0.90) -> None:
        self.core_floor = float(core_floor)

    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 occlusion_allowed_hair: torch.Tensor,
                 target_hair_core: torch.Tensor, target_hair_transition: torch.Tensor,
                 anchor_hair_evidence: torch.Tensor, face_contact_ring: torch.Tensor,
                 face_intrusion_risk: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, return_aux: bool = False, **_: torch.Tensor):
        del source_face_mask, source_skin_mask
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        core = target_hair_core.float().clamp(0, 1)
        transition = target_hair_transition.float().clamp(0, 1)
        evidence = anchor_hair_evidence.float().clamp(0, 1)
        ring = face_contact_ring.float().clamp(0, 1)
        risk = face_intrusion_risk.float().clamp(0, 1)
        boundary_alpha = (0.65 * evidence + 0.20 * F.avg_pool2d(coarse, 5, 1, 2) +
                          0.15 * F.avg_pool2d(evidence, 5, 1, 2)).clamp(0, 1)
        boundary_alpha = (boundary_alpha * (1.0 - 0.75 * ring * risk)).clamp(0, 1)
        alpha = (core * self.core_floor + transition * boundary_alpha).clamp(0, 1)
        alpha = torch.where(core > 0.5, torch.maximum(alpha, core * self.core_floor), alpha)
        alpha = alpha * (coarse > 1e-4).float()
        aux = {
            "target_hair_alpha_final": alpha,
            "target_hair_core": core,
            "target_hair_transition": transition,
            "target_hair_background": (coarse <= 0.02).float(),
            "boundary_confidence": boundary_alpha,
            "boundary_artifact_map": (alpha - F.avg_pool2d(alpha, 3, 1, 1)).abs(),
        }
        return (alpha, aux) if return_aux else alpha


__all__ = ["TargetHairMatteRefinerV842"]
