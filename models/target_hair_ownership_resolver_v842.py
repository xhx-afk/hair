"""V2.42 topology-first target hair ownership."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


class TargetHairOwnershipResolverV842:
    """Keep target topology authoritative and use evidence only on transitions."""

    def __init__(self, *, contact_radius: int = 5, core_radius: int = 5) -> None:
        self.contact_radius = max(int(contact_radius), 1)
        self.core_radius = max(int(core_radius), 1)

    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 source_hair_mask: torch.Tensor | None = None, source_face_mask: torch.Tensor | None = None,
                 source_skin_mask: torch.Tensor | None = None, anchor_hair_evidence: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        target = coarse_target_hair_mask.float().clamp(0, 1)
        source_hair = torch.zeros_like(target) if source_hair_mask is None else source_hair_mask.float().clamp(0, 1)
        face = torch.zeros_like(target) if source_face_mask is None else source_face_mask.float().clamp(0, 1)
        skin = torch.zeros_like(target) if source_skin_mask is None else source_skin_mask.float().clamp(0, 1)
        evidence = torch.ones_like(target) if anchor_hair_evidence is None else anchor_hair_evidence.float().clamp(0, 1)
        core = _erode(target, self.core_radius).clamp(0, 1)
        transition = (target - core).clamp(0, 1)
        topology = F.avg_pool2d(target, 9, stride=1, padding=4).clamp(0, 1)
        connectivity = F.avg_pool2d(target, 5, stride=1, padding=2).clamp(0, 1)
        transition_confidence = (0.65 * evidence + 0.20 * topology + 0.15 * connectivity).clamp(0, 1)
        core_ownership = core
        transition_ownership = (transition * transition_confidence).clamp(0, 1)
        ownership = (core_ownership + transition_ownership).clamp(0, 1)
        new_region = (target * (1.0 - source_hair)).clamp(0, 1)
        new_core = (core * (1.0 - source_hair)).clamp(0, 1)
        new_transition = (transition * (1.0 - source_hair)).clamp(0, 1)
        contact_band = (_dilate(target, self.contact_radius) - _erode(target, self.contact_radius)).clamp(0, 1)
        overlap = target * face
        skin_contact = target * skin
        face_intrusion_risk = (transition * face * (1.0 - transition_confidence) *
                               (0.5 + 0.5 * skin)).clamp(0, 1)
        aux = {
            "coarse_target_hair": target,
            "target_hair_mask": target,
            "source_hair_mask": source_hair,
            "target_hair_core": core,
            "target_hair_transition": transition,
            "target_core_ownership": core_ownership,
            "target_transition_ownership": transition_ownership,
            "target_hair_ownership": ownership,
            "new_hair_region": new_region,
            "new_hair_core": new_core,
            "new_hair_transition": new_transition,
            "transition_confidence": transition_confidence,
            "anchor_hair_evidence": evidence,
            "topology_connectedness": connectivity,
            "overlap_face_hair": overlap,
            "face_overlap": overlap,
            "skin_contact": skin_contact,
            "face_contact_ring": contact_band * (face + skin).clamp(0, 1),
            "face_intrusion_risk": face_intrusion_risk,
            "deep_overlap": transition * skin * (1.0 - contact_band),
            "occlusion_allowed_hair": ownership,
        }
        return (ownership, aux) if return_aux else ownership


__all__ = ["TargetHairOwnershipResolverV842"]
