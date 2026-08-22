"""Unified, geometry-independent hair evidence for matte and face guard."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _robust_normalize(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = []
    for batch in range(value.size(0)):
        pixels = value[batch, 0][mask[batch, 0] > 0.05]
        if pixels.numel() < 8:
            pixels = value[batch, 0].flatten()
        q50, q90 = torch.quantile(pixels, value.new_tensor([0.50, 0.90]))
        span = q90 - q50
        if span < 1e-4:
            normalized = value[batch:batch + 1] / q90.clamp_min(1e-4)
        else:
            normalized = (value[batch:batch + 1] - q50) / span
        output.append(normalized.clamp(0, 1))
    return torch.cat(output, dim=0)


class AnchorHairEvidenceV8411:
    def __call__(self, *, allowed_hair: torch.Tensor,
                 coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                 source_skin_mask: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 base_rgb: torch.Tensor, face_contact_ring: torch.Tensor | None = None,
                 return_aux: bool = False):
        del source_skin_mask, face_contact_ring
        allowed = allowed_hair.float().clamp(0, 1)
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        face = source_face_mask.float().clamp(0, 1)
        overlap = face * coarse
        evidence_mask = (allowed + coarse).clamp(0, 1)
        texture = (strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 3)).abs().mean(1, keepdim=True)
        anchor_delta = (strong_anchor_rgb - base_rgb).abs().mean(1, keepdim=True)
        texture_only_conf = _robust_normalize(texture, evidence_mask)
        delta_conf = _robust_normalize(anchor_delta, evidence_mask)
        # Appearance delta can be caused by non-hair drift. In face overlap
        # it contributes at most 7.5%; independent strand texture dominates.
        delta_weight = 0.25 * (1.0 - 0.70 * overlap)
        texture_weight = 1.0 - delta_weight
        texture_conf = (texture_weight * texture_only_conf + delta_weight * delta_conf).clamp(0, 1)
        local_structure = F.avg_pool2d(texture_conf, 5, stride=1, padding=2)
        topology_support = F.avg_pool2d(coarse, 9, stride=1, padding=4)
        topology_weight = 0.30 * (1.0 - overlap) + 0.10 * overlap
        topology_conf = topology_weight * topology_support
        evidence = (0.70 * texture_conf + 0.20 * local_structure + topology_conf).clamp(0, 1)
        aux = {
            "anchor_texture": texture,
            "anchor_base_delta": anchor_delta,
            "anchor_texture_only_confidence": texture_only_conf,
            "anchor_texture_confidence": texture_conf,
            "local_structure_confidence": local_structure,
            "topology_support": topology_support,
            "anchor_hair_evidence": evidence,
        }
        return (evidence, aux) if return_aux else evidence


__all__ = ["AnchorHairEvidenceV8411"]
