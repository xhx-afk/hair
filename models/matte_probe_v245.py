"""Continuous matte probe used only by the V2.45 Matte Death Test."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v245_death_test_common import dilate, smoothstep


class MatteProbeV245:
    def __init__(self, *, support_radius: int = 4, flyaway_cap: float = 0.45) -> None:
        self.support_radius = int(support_radius)
        self.flyaway_cap = float(flyaway_cap)

    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 distance_prior: torch.Tensor, anchor_hair_evidence: torch.Tensor,
                 strand_structure_confidence: torch.Tensor,
                 source_skin_mask: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                 return_aux: bool = False):
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        distance = distance_prior.float().clamp(0, 1)
        evidence = anchor_hair_evidence.float().clamp(0, 1)
        strand = strand_structure_confidence.float().clamp(0, 1)
        skin = source_skin_mask.float().clamp(0, 1)
        inside_confidence = 0.35 * distance + 0.40 * evidence + 0.25 * strand
        skin_risk = skin * (1.0 - evidence) * (1.0 - strand)
        inside_alpha = coarse * inside_confidence * (1.0 - 0.55 * skin_risk)
        interior_boost = smoothstep(distance, 0.35, 0.75)
        inside_alpha = (inside_alpha + interior_boost * evidence * (1.0 - inside_alpha) * 0.55).clamp(0, 1)

        support = dilate(coarse, self.support_radius)
        outside = support * (1.0 - coarse)
        local_mean = F.avg_pool2d(strong_anchor_rgb.float(), 5, stride=1, padding=2)
        local_contrast = (strong_anchor_rgb.float() - local_mean).abs().mean(1, keepdim=True)
        contrast_scale = local_contrast / local_contrast.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
        contrast_scale = contrast_scale.clamp(0, 1)
        flyaway_score = (outside * strand * contrast_scale * (1.0 - 0.65 * skin_risk)).clamp(0, 1)
        flyaway_alpha = (self.flyaway_cap * flyaway_score).clamp(0, self.flyaway_cap)
        alpha = (1.0 - (1.0 - inside_alpha) * (1.0 - flyaway_alpha)).clamp(0, 1)
        aux = {
            "inside_confidence": inside_confidence,
            "skin_risk": skin_risk,
            "inside_alpha": inside_alpha,
            "interior_boost": interior_boost,
            "hair_support": support,
            "outside": outside,
            "flyaway_score": flyaway_score,
            "flyaway_alpha": flyaway_alpha,
            "independent_flyaway_candidate": (flyaway_score > 0.45).float() * outside,
        }
        return (alpha, aux) if return_aux else alpha


__all__ = ["MatteProbeV245"]
