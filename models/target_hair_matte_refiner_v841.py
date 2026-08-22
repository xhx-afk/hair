"""V2.41 matte refinement with anchor evidence in face overlap regions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


class TargetHairMatteRefinerV841:
    def __init__(self, *, ring_radius: int = 5) -> None:
        self.ring_radius = max(int(ring_radius), 1)

    def __call__(self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 new_hair_rgb: torch.Tensor, coarse_target_hair_mask: torch.Tensor,
                 occlusion_allowed_hair: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, face_contact_ring: torch.Tensor,
                 face_intrusion_risk: torch.Tensor, anchor_hair_evidence: torch.Tensor | None = None,
                 anchor_texture_confidence: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        del base_rgb, strong_anchor_rgb, new_hair_rgb, source_skin_mask
        if anchor_hair_evidence is None:
            raise ValueError("V2.41.1 requires unified anchor_hair_evidence")
        allowed = occlusion_allowed_hair.float().clamp(0, 1)
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        core = _erode(allowed, self.ring_radius).clamp(0, 1)
        support = _dilate(allowed, 1).clamp(0, 1)
        transition = (allowed - core).clamp(0, 1)
        texture_conf = anchor_hair_evidence.float().clamp(0, 1) if anchor_texture_confidence is None else anchor_texture_confidence.float().clamp(0, 1)
        overlap = (source_face_mask.float().clamp(0, 1) * coarse).clamp(0, 1)
        geometry_weight = 0.55 * (1.0 - 0.5 * overlap)
        evidence_weight = (1.0 - geometry_weight - 0.25).clamp_min(0.0)
        boundary_alpha = (geometry_weight * F.avg_pool2d(allowed, 5, stride=1, padding=2) +
                          0.25 * texture_conf + evidence_weight * anchor_hair_evidence).clamp(0, 1)
        boundary_alpha = boundary_alpha * (1.0 - face_contact_ring.float().clamp(0, 1) * face_intrusion_risk.float().clamp(0, 1)).clamp(0, 1)
        alpha = (core + transition * boundary_alpha).clamp(0, 1) * support
        alpha = torch.where(core > 0.95, torch.ones_like(alpha), alpha)
        alpha = torch.where(allowed <= 1e-4, torch.zeros_like(alpha), alpha)
        if not torch.isfinite(alpha).all():
            raise ValueError("V2.41 matte refiner produced NaN or Inf")
        if not return_aux:
            return alpha
        return alpha, {
            "target_hair_alpha_final": alpha,
            "target_hair_core": (alpha >= 0.85).float(),
            "target_hair_transition": ((alpha > 0.02) & (alpha < 0.85)).float(),
            "target_hair_background": (alpha <= 0.02).float(),
            "boundary_confidence": boundary_alpha,
            "boundary_geometry_weight": geometry_weight,
            "boundary_anchor_texture_confidence": texture_conf,
            "boundary_anchor_hair_evidence": anchor_hair_evidence,
            "boundary_artifact_map": (alpha - F.avg_pool2d(alpha, 3, stride=1, padding=1)).abs(),
        }


__all__ = ["TargetHairMatteRefinerV841"]
