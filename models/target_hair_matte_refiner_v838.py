"""V2.38 high-resolution rule-based target hair matte refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


class TargetHairMatteRefinerV838:
    def __init__(self, *, ring_radius: int = 5, face_guard: float = 0.35,
                 use_appearance_confidence: bool = True) -> None:
        if ring_radius < 1 or not 0 <= face_guard <= 1:
            raise ValueError("V2.38 matte refiner parameters are invalid")
        self.ring_radius = int(ring_radius)
        self.face_guard = float(face_guard)
        self.use_appearance_confidence = bool(use_appearance_confidence)

    def __call__(
        self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
        new_hair_rgb: torch.Tensor, coarse_target_hair_mask: torch.Tensor,
        occlusion_allowed_hair: torch.Tensor, source_face_mask: torch.Tensor,
        source_skin_mask: torch.Tensor, face_contact_ring: torch.Tensor,
        face_intrusion_risk: torch.Tensor, return_aux: bool = False,
    ):
        allowed = occlusion_allowed_hair.float().clamp(0, 1)
        core = _erode(allowed, self.ring_radius).clamp(0, 1)
        support = _dilate(allowed, 1).clamp(0, 1)
        transition = (allowed - core).clamp(0, 1)
        appearance = (new_hair_rgb - base_rgb).abs().mean(1, keepdim=True)
        texture = (strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 3)).abs().mean(1, keepdim=True)
        appearance_confidence = (appearance / (appearance.mean(dim=(-2, -1), keepdim=True) + 1e-4)).clamp(0, 2) / 2
        texture_confidence = (texture / (texture.mean(dim=(-2, -1), keepdim=True) + 1e-4)).clamp(0, 2) / 2
        geometry_confidence = F.avg_pool2d(allowed, 5, stride=1, padding=2)
        if self.use_appearance_confidence:
            boundary_alpha = (0.45 * geometry_confidence + 0.30 * appearance_confidence + 0.25 * texture_confidence).clamp(0, 1)
        else:
            boundary_alpha = (0.60 * geometry_confidence + 0.25 * texture_confidence + 0.15 * geometry_confidence).clamp(0, 1)
        boundary_alpha = boundary_alpha * (1.0 - self.face_guard * face_contact_ring * face_intrusion_risk)
        alpha = (core + transition * boundary_alpha).clamp(0, 1) * support
        alpha = torch.where(core > 0.95, torch.ones_like(alpha), alpha)
        alpha = torch.where(allowed <= 1e-4, torch.zeros_like(alpha), alpha)
        core_final = (alpha >= 0.85).float()
        transition_final = ((alpha > 0.02) & (alpha < 0.85)).float()
        background = (alpha <= 0.02).float()
        if not torch.isfinite(alpha).all():
            raise ValueError("V2.38 matte refiner produced NaN or Inf")
        if not return_aux:
            return alpha
        return alpha, {
            "target_hair_alpha_final": alpha,
            "target_hair_core": core_final,
            "target_hair_transition": transition_final,
            "target_hair_background": background,
            "boundary_confidence": boundary_alpha,
            "boundary_artifact_map": (alpha - F.avg_pool2d(alpha, 3, stride=1, padding=1)).abs(),
        }


__all__ = ["TargetHairMatteRefinerV838"]
