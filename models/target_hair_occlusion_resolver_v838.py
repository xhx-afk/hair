"""V2.38 target-hair ownership that allows plausible hair-over-face occlusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _soft_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _soft_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


class TargetHairOcclusionResolverV838:
    """Use target topology and appearance evidence instead of source-face veto."""

    def __init__(self, *, contact_radius: int = 5, risk_strength: float = 0.65) -> None:
        if contact_radius < 1 or not 0 <= risk_strength <= 1:
            raise ValueError("V2.38 occlusion resolver parameters are invalid")
        self.contact_radius = int(contact_radius)
        self.risk_strength = float(risk_strength)

    def __call__(
        self, *, coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
        source_skin_mask: torch.Tensor, strong_anchor_rgb: torch.Tensor,
        base_rgb: torch.Tensor, return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        face = source_face_mask.float().clamp(0, 1)
        skin = source_skin_mask.float().clamp(0, 1)
        overlap = coarse * face
        contact_band = (_soft_dilate(coarse, self.contact_radius) - _soft_erode(coarse, self.contact_radius)).clamp(0, 1)
        anchor_hf = strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 3)
        base_hf = base_rgb - gaussian_blur_v829(base_rgb, 3)
        texture_evidence = (anchor_hf - base_hf).abs().mean(1, keepdim=True)
        local_texture = F.avg_pool2d(texture_evidence, 7, stride=1, padding=3)
        texture_confidence = (local_texture / (local_texture.mean(dim=(-2, -1), keepdim=True) + 1e-4)).clamp(0, 2) / 2
        connectedness = F.avg_pool2d(coarse, 9, stride=1, padding=4)
        # High target-hair texture and connectedness permit hair-over-face; skin
        # only suppresses uncertain contact pixels instead of vetoing all overlap.
        allowed_overlap = (0.55 * texture_confidence + 0.45 * connectedness).clamp(0, 1)
        face_intrusion_risk = (overlap * (1.0 - allowed_overlap) * (0.5 + 0.5 * skin)).clamp(0, 1)
        allowed = (coarse * (1.0 - self.risk_strength * face_intrusion_risk)).clamp(0, 1)
        allowed = torch.where(overlap > 0, torch.maximum(allowed, overlap * allowed_overlap), allowed)
        if not return_aux:
            return allowed
        return allowed, {
            "coarse_target_hair": coarse,
            "overlap_face_hair": overlap,
            "face_contact_ring": contact_band * (face + skin).clamp(0, 1),
            "face_intrusion_risk": face_intrusion_risk,
            "texture_confidence": texture_confidence,
            "topology_connectedness": connectedness,
            "occlusion_allowed_hair": allowed,
        }


__all__ = ["TargetHairOcclusionResolverV838"]
