"""Contact-only contamination guard independent of RGB appearance delta."""

from __future__ import annotations

import torch

from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


class FaceContactContaminationGuardV840:
    def __init__(self, *, strength: float = 0.85, hard_threshold: float = 0.80) -> None:
        self.strength, self.hard_threshold = float(strength), float(hard_threshold)

    def __call__(self, *, alpha: torch.Tensor, allowed_hair: torch.Tensor,
                 source_face_mask: torch.Tensor, source_skin_mask: torch.Tensor,
                 face_contact_ring: torch.Tensor, face_intrusion_risk: torch.Tensor,
                 strong_anchor_rgb: torch.Tensor, base_rgb: torch.Tensor,
                 new_hair_rgb: torch.Tensor, hair_core: torch.Tensor,
                 return_aux: bool = False):
        del source_skin_mask, new_hair_rgb
        anchor_delta = (strong_anchor_rgb - base_rgb).abs().mean(1, keepdim=True)
        anchor_texture = (strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 3)).abs().mean(1, keepdim=True)
        anchor_evidence = (0.55 * anchor_delta + 0.45 * anchor_texture)
        anchor_evidence = (anchor_evidence / (anchor_evidence.mean(dim=(-2, -1), keepdim=True) + 1e-4)).clamp(0, 2) / 2
        contact = (face_contact_ring * source_face_mask.float().clamp(0, 1)).clamp(0, 1)
        delta_norm = ((strong_anchor_rgb - base_rgb).abs().mean(1, keepdim=True) / 0.35).clamp(0, 1)
        risk = (contact * face_intrusion_risk * (1.0 - anchor_evidence) * delta_norm).clamp(0, 1)
        guarded = (alpha * (1.0 - self.strength * risk)).clamp(0, 1)
        hard_reject = (risk > self.hard_threshold) & (hair_core <= 0.5)
        guarded = torch.where(hard_reject, torch.minimum(guarded, torch.full_like(guarded, 0.05)), guarded)
        guarded = torch.where(allowed_hair <= 1e-4, torch.zeros_like(guarded), guarded)
        if not return_aux:
            return guarded
        return guarded, {
            "anchor_hair_evidence": anchor_evidence,
            "face_contact_contamination_risk": risk,
            "hard_contact_reject": hard_reject.float(),
            "hair_alpha_guarded": guarded,
        }


__all__ = ["FaceContactContaminationGuardV840"]
