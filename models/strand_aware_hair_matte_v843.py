"""V2.43 strand-aware alpha construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    return F.max_pool2d(mask.float().clamp(0, 1), 2 * radius + 1, stride=1, padding=radius)


class StrandAwareHairMatteV843:
    def __init__(self, *, support_radius: int = 4, safe_alpha: float = 0.92,
                 flyaway_alpha_cap: float = 0.55) -> None:
        self.support_radius = int(support_radius)
        self.safe_alpha = float(safe_alpha)
        self.flyaway_alpha_cap = float(flyaway_alpha_cap)

    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 safe_dense_core: torch.Tensor, uncertain_core: torch.Tensor,
                 transition_prior: torch.Tensor | None = None,
                 anchor_hair_evidence: torch.Tensor,
                 strand_structure_confidence: torch.Tensor,
                 local_connectivity: torch.Tensor | None = None,
                 source_skin_mask: torch.Tensor | None = None,
                 skin_conflict: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        coarse = coarse_target_hair_mask.float().clamp(0, 1)
        safe = safe_dense_core.float().clamp(0, 1)
        uncertain = uncertain_core.float().clamp(0, 1)
        evidence = anchor_hair_evidence.float().clamp(0, 1)
        strand = strand_structure_confidence.float().clamp(0, 1)
        skin = torch.zeros_like(coarse) if source_skin_mask is None else source_skin_mask.float().clamp(0, 1)
        conflict = torch.zeros_like(coarse) if skin_conflict is None else skin_conflict.float().clamp(0, 1)
        connectivity = F.avg_pool2d(coarse, 5, stride=1, padding=2) if local_connectivity is None else local_connectivity.float().clamp(0, 1)
        target_transition = (coarse - safe).clamp(0, 1) if transition_prior is None else transition_prior.float().clamp(0, 1)
        quality = (0.45 * evidence + 0.35 * strand + 0.20 * connectivity).clamp(0, 1)
        uncertain_alpha = (quality * uncertain * (1.0 - 0.60 * conflict)).clamp(0, 1)
        transition_alpha = (target_transition * quality).clamp(0, 1)
        support = _dilate(coarse, self.support_radius)
        outside_band = (support * (1.0 - coarse)).clamp(0, 1)
        non_skin_candidate = ((strand > 0.65) & (evidence > 0.55)).float()
        skin_candidate = ((strand > 0.78) & (evidence > 0.70)).float()
        threshold = torch.where(skin > 0.5, skin_candidate, non_skin_candidate)
        flyaway_candidate = outside_band * threshold
        flyaway_alpha = (outside_band * strand * evidence * 0.55 * threshold).clamp(0, self.flyaway_alpha_cap)
        safe_alpha = (safe * self.safe_alpha).clamp(0, 1)
        alpha = 1.0 - (1.0 - safe_alpha) * (1.0 - uncertain_alpha) * (1.0 - transition_alpha) * (1.0 - flyaway_alpha)
        alpha = alpha.clamp(0, 1)
        aux = {
            "safe_core_alpha": safe_alpha,
            "uncertain_core_alpha": uncertain_alpha,
            "transition_alpha": transition_alpha,
            "hair_support": support,
            "outside_band": outside_band,
            "flyaway_candidate": flyaway_candidate,
            "flyaway_alpha": flyaway_alpha,
            "target_hair_alpha_final": alpha,
            "local_connectivity": connectivity,
            "alpha_before_guard": alpha,
        }
        return (alpha, aux) if return_aux else alpha


__all__ = ["StrandAwareHairMatteV843"]
