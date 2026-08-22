"""V2.41 full face/skin overlap contamination guard."""

from __future__ import annotations

import torch

def _robust_normalize(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = []
    for batch in range(value.size(0)):
        pixels = value[batch, 0][mask[batch, 0] > 0.05]
        if pixels.numel() < 8:
            pixels = value[batch, 0].flatten()
        q50, q90 = torch.quantile(pixels, value.new_tensor([0.50, 0.90]))
        span = q90 - q50
        normalized = value[batch:batch + 1] / q90.clamp_min(1e-4) if span < 1e-4 else (value[batch:batch + 1] - q50) / span
        out.append(normalized.clamp(0, 1))
    return torch.cat(out, dim=0)


class FaceOverlapContaminationGuardV841:
    def __init__(self, *, strength: float = 0.85) -> None:
        self.strength = float(strength)

    def __call__(self, *, alpha: torch.Tensor, allowed_hair: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 base_rgb: torch.Tensor, new_hair_rgb: torch.Tensor,
                 face_contact_ring: torch.Tensor, face_intrusion_risk: torch.Tensor,
                 hair_core: torch.Tensor, anchor_hair_evidence: torch.Tensor,
                 return_aux: bool = False):
        allowed = allowed_hair.float().clamp(0, 1)
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        face = source_face_mask.float().clamp(0, 1)
        skin = source_skin_mask.float().clamp(0, 1)
        ring = face_contact_ring.float().clamp(0, 1)
        intrusion = face_intrusion_risk.float().clamp(0, 1)
        del strong_anchor_rgb
        hair_evidence = anchor_hair_evidence.float().clamp(0, 1)

        recolor_delta = (new_hair_rgb - base_rgb).abs().mean(1, keepdim=True)
        recolor_delta_norm = _robust_normalize(recolor_delta, coarse)
        skin_contact = skin * coarse
        face_overlap = face * coarse
        deep_overlap = skin_contact * (1.0 - ring)
        overlap_weight = torch.maximum(face_overlap, skin_contact)
        contamination_risk = (overlap_weight * intrusion * (1.0 - hair_evidence) * recolor_delta_norm).clamp(0, 1)
        contact_risk = (ring * skin * intrusion * (1.0 - hair_evidence)).clamp(0, 1)
        combined_risk = torch.maximum(contamination_risk, contact_risk)
        guard_strength_map = self.strength * (1.0 - 0.65 * hair_core.float().clamp(0, 1))
        guarded = (alpha.float().clamp(0, 1) * (1.0 - guard_strength_map * combined_risk)).clamp(0, 1)
        # Deep skin overlap is strongly suppressed only when there is no
        # independent hair evidence; a confident bang remains legal over face.
        deep_invalid = deep_overlap * (hair_evidence < 0.25).float() * (intrusion > 0.30).float()
        deep_uncertain = deep_overlap * (hair_evidence >= 0.25).float() * (hair_evidence < 0.50).float()
        guarded = torch.where(deep_uncertain > 0, guarded * 0.25, guarded)
        guarded = torch.where(deep_invalid > 0, torch.minimum(guarded, torch.full_like(guarded, 0.05)), guarded)
        guarded = torch.where(allowed <= 1e-4, torch.zeros_like(guarded), guarded)
        if not return_aux:
            return guarded
        return guarded, {
            "anchor_hair_evidence": hair_evidence,
            "recolor_delta": recolor_delta,
            "recolor_delta_norm": recolor_delta_norm,
            "skin_contact": skin_contact,
            "skin_overlap": skin_contact,
            "face_overlap": face_overlap,
            "deep_overlap": deep_overlap,
            "deep_skin_overlap": deep_overlap,
            "face_overlap_contamination_risk": contamination_risk,
            "face_contact_contamination_risk": contact_risk,
            "combined_contamination_risk": combined_risk,
            "deep_invalid": deep_invalid,
            "deep_uncertain": deep_uncertain,
            "hair_alpha_guarded": guarded,
            "confident_hair_evidence": hair_evidence,
            "uncertain_skin_overlap": skin_contact * (1.0 - hair_evidence),
        }


__all__ = ["FaceOverlapContaminationGuardV841"]
