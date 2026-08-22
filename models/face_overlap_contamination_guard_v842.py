"""V2.42 face guard restricted to target hair transitions."""

from __future__ import annotations

import torch


class FaceOverlapContaminationGuardV842:
    def __init__(self, *, strength: float = 0.85) -> None:
        self.strength = float(strength)

    def __call__(self, *, alpha: torch.Tensor, allowed_hair: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, face_contact_ring: torch.Tensor,
                 face_intrusion_risk: torch.Tensor, hair_core: torch.Tensor,
                 anchor_hair_evidence: torch.Tensor, new_hair_core: torch.Tensor | None = None,
                 base_rgb: torch.Tensor | None = None, new_hair_rgb: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        del allowed_hair
        target = coarse_target_hair_mask.float().clamp(0, 1)
        face = source_face_mask.float().clamp(0, 1)
        skin = source_skin_mask.float().clamp(0, 1)
        transition = (target - hair_core.float().clamp(0, 1)).clamp(0, 1)
        evidence = anchor_hair_evidence.float().clamp(0, 1)
        risk = face_intrusion_risk.float().clamp(0, 1)
        guard_zone = transition * skin
        if new_hair_core is not None:
            guard_zone = guard_zone * (1.0 - new_hair_core.float().clamp(0, 1))
        deep_invalid = guard_zone * (evidence < 0.25).float() * (risk > 0.30).float()
        deep_uncertain = guard_zone * (evidence >= 0.25).float() * (evidence < 0.50).float()
        combined = (guard_zone * risk * (1.0 - evidence)).clamp(0, 1)
        guarded = alpha.float().clamp(0, 1) * (1.0 - self.strength * combined)
        guarded = torch.where(deep_uncertain > 0, guarded * 0.25, guarded)
        guarded = torch.where(deep_invalid > 0, torch.minimum(guarded, torch.full_like(guarded, 0.05)), guarded)
        guarded = guarded.clamp(0, 1)
        if base_rgb is not None and new_hair_rgb is not None:
            recolor_delta = (new_hair_rgb.float() - base_rgb.float()).abs().mean(1, keepdim=True)
            recolor_delta_norm = recolor_delta / recolor_delta.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        else:
            recolor_delta = torch.zeros_like(alpha)
            recolor_delta_norm = torch.zeros_like(alpha)
        aux = {
            "anchor_hair_evidence": evidence, "face_overlap": target * face,
            "skin_contact": target * skin, "deep_overlap": guard_zone,
            "guard_zone": guard_zone, "deep_invalid": deep_invalid,
            "deep_uncertain": deep_uncertain, "face_overlap_contamination_risk": combined,
            "face_contact_contamination_risk": combined, "combined_contamination_risk": combined,
            "hair_alpha_guarded": guarded,
            # Keep the v2.41 diagnostic contract while the v2.42 guard remains
            # transition-only. These maps are descriptive, not gating inputs.
            "recolor_delta": recolor_delta,
            "recolor_delta_norm": recolor_delta_norm,
            "skin_overlap": target * skin,
            "deep_skin_overlap": guard_zone,
            "confident_hair_evidence": evidence,
            "uncertain_skin_overlap": target * skin * (1.0 - evidence),
        }
        return (guarded, aux) if return_aux else guarded


__all__ = ["FaceOverlapContaminationGuardV842"]
