"""V2.41 ownership resolver facade with explicit overlap diagnostics."""

from __future__ import annotations

import torch

from models.target_hair_occlusion_resolver_v838 import TargetHairOcclusionResolverV838


class TargetHairOcclusionResolverV841(TargetHairOcclusionResolverV838):
    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 source_face_mask: torch.Tensor, source_skin_mask: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor, base_rgb: torch.Tensor,
                 return_aux: bool = False):
        result = super().__call__(
            coarse_target_hair_mask=coarse_target_hair_mask,
            source_face_mask=source_face_mask, source_skin_mask=source_skin_mask,
            strong_anchor_rgb=strong_anchor_rgb, base_rgb=base_rgb, return_aux=True,
        )
        allowed, aux = result
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        face = source_face_mask.float().clamp(0, 1)
        skin = source_skin_mask.float().clamp(0, 1)
        ring = aux["face_contact_ring"].float().clamp(0, 1)
        skin_contact = skin * coarse
        face_overlap = face * coarse
        aux.update({
            "skin_contact": skin_contact,
            "face_overlap": face_overlap,
            "deep_overlap": skin_contact * (1.0 - ring),
        })
        return (allowed, aux) if return_aux else allowed


__all__ = ["TargetHairOcclusionResolverV841"]
