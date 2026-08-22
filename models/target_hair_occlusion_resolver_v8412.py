"""V2.41.2 evidence-first target hair ownership resolver."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _soft_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _soft_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


class TargetHairOcclusionResolverV8412:
    """Resolve face overlap from independent hair evidence before matting."""

    def __init__(self, *, contact_radius: int = 5) -> None:
        if contact_radius < 1:
            raise ValueError("V2.41.2 occlusion resolver parameters are invalid")
        self.contact_radius = int(contact_radius)

    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 source_face_mask: torch.Tensor, source_skin_mask: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor, base_rgb: torch.Tensor,
                 anchor_hair_evidence: torch.Tensor, source_hair_mask: torch.Tensor | None = None,
                 return_aux: bool = False):
        del source_hair_mask
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        face = source_face_mask.float().clamp(0, 1)
        skin = source_skin_mask.float().clamp(0, 1)
        evidence = anchor_hair_evidence.float().clamp(0, 1)
        overlap = coarse * face
        skin_contact = coarse * skin
        contact_band = (_soft_dilate(coarse, self.contact_radius) -
                        _soft_erode(coarse, self.contact_radius)).clamp(0, 1)
        anchor_hf = strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 3)
        base_hf = base_rgb - gaussian_blur_v829(base_rgb, 3)
        texture = (anchor_hf - base_hf).abs().mean(1, keepdim=True)
        local_texture = F.avg_pool2d(texture, 7, stride=1, padding=3)
        texture_support = (local_texture / (local_texture.mean(dim=(-2, -1), keepdim=True) + 1e-4)).clamp(0, 2) / 2
        topology = F.avg_pool2d(coarse, 9, stride=1, padding=4)
        continuity = F.avg_pool2d(evidence, 5, stride=1, padding=2)
        overlap_confidence = (0.75 * evidence + 0.15 * topology + 0.10 * continuity).clamp(0, 1)
        allowed_overlap = overlap * overlap_confidence
        face_intrusion_risk = (overlap * (1.0 - overlap_confidence) *
                               (0.5 + 0.5 * skin)).clamp(0, 1)
        # Outside face overlap the coarse prior remains the support. Inside
        # overlap, ownership is explicitly evidence-weighted rather than
        # rescued by connectedness or a geometry-derived fallback.
        allowed = torch.where(overlap > 0, allowed_overlap, coarse).clamp(0, 1)
        if not return_aux:
            return allowed
        return allowed, {
            "coarse_target_hair": coarse,
            "overlap_face_hair": overlap,
            "face_overlap": overlap,
            "skin_contact": skin_contact,
            "face_contact_ring": contact_band * (face + skin).clamp(0, 1),
            "face_intrusion_risk": face_intrusion_risk,
            "texture_confidence": texture_support,
            "topology_connectedness": topology,
            "anchor_hair_evidence": evidence,
            "overlap_confidence": overlap_confidence,
            "deep_overlap": skin_contact * (1.0 - contact_band * (face + skin).clamp(0, 1)),
            "occlusion_allowed_hair": allowed,
        }


__all__ = ["TargetHairOcclusionResolverV8412"]
