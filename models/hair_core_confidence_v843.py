"""V2.43 confidence-core construction for target hair mattes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hair_strand_structure_v843 import strand_structure_confidence_v843


def _erode(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(int(radius), 1)
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _dilate(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(int(radius), 1)
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def _resize(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask


def distance_prior_v843(mask: torch.Tensor, *, steps: int = 12) -> torch.Tensor:
    """Approximate continuous distance-to-boundary prior using soft erosions."""
    target = mask.float().clamp(0, 1)
    current = target
    accumulated = torch.zeros_like(target)
    for _ in range(max(int(steps), 1)):
        current = _erode(current)
        accumulated = accumulated + current.clamp(0, 1)
    # A broad mask should retain a broad prior, while a thin boundary should
    # quickly decay.  This remains a prior; confidence gates it below.
    prior = accumulated / max(float(steps), 1.0)
    # Normalize per sample so the deepest connected hair receives a meaningful
    # prior even when the target mask is wider than the configured step count.
    peak = prior.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (prior / peak).clamp(0, 1)


class HairCoreConfidenceV843:
    def __init__(self, *, distance_steps: int = 12, strand_blur_radius: int = 2,
                 safe_threshold: float = 0.72, uncertain_threshold: float = 0.40) -> None:
        self.distance_steps = int(distance_steps)
        self.strand_blur_radius = int(strand_blur_radius)
        self.safe_threshold = float(safe_threshold)
        self.uncertain_threshold = float(uncertain_threshold)

    def __call__(self, *, coarse_target_hair_mask: torch.Tensor,
                 source_hair_mask: torch.Tensor | None = None,
                 source_face_mask: torch.Tensor | None = None,
                 source_skin_mask: torch.Tensor | None = None,
                 strong_anchor_rgb: torch.Tensor | None = None,
                 base_rgb: torch.Tensor | None = None,
                 anchor_hair_evidence: torch.Tensor | None = None,
                 strand_structure_confidence: torch.Tensor | None = None,
                 anchor_l: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        del base_rgb, source_face_mask
        target = coarse_target_hair_mask.float().clamp(0, 1)
        size = target.shape[-2:]
        source_hair = torch.zeros_like(target) if source_hair_mask is None else _resize(source_hair_mask, size)
        skin = torch.zeros_like(target) if source_skin_mask is None else _resize(source_skin_mask, size)
        if anchor_l is None:
            if strong_anchor_rgb is None:
                anchor_l = target * 50.0
            else:
                anchor_l = rgb_to_lab(strong_anchor_rgb)[:, :1]
        anchor_l = _resize(anchor_l, size)
        evidence = torch.ones_like(target) if anchor_hair_evidence is None else _resize(anchor_hair_evidence, size)
        support = _dilate(target, 2)
        distance = distance_prior_v843(target, steps=self.distance_steps)
        strand_aux = strand_structure_confidence_v843(
            anchor_l, local_hair_support=support, anchor_hair_evidence=evidence,
            blur_radius=self.strand_blur_radius,
        )
        strand = strand_aux["strand_structure_confidence"] if strand_structure_confidence is None else _resize(strand_structure_confidence, size)
        strand_aux["strand_structure_confidence"] = strand
        skin_conflict = (skin * (1.0 - evidence) * (1.0 - strand)).clamp(0, 1)
        dense = (0.35 * distance + 0.35 * evidence + 0.30 * strand).clamp(0, 1)
        dense = (dense * (1.0 - 0.40 * skin_conflict)).clamp(0, 1)
        safe = (dense >= self.safe_threshold).float()
        uncertain = ((dense >= self.uncertain_threshold) & (dense < self.safe_threshold)).float()
        transition = (target - safe).clamp(0, 1)
        out = {
            "distance_prior": distance,
            "strand_structure_confidence": strand,
            "skin_conflict": skin_conflict,
            "dense_core_confidence": dense,
            "safe_dense_core": safe,
            "uncertain_core": uncertain,
            "transition_prior": transition,
            "anchor_hair_evidence": evidence,
            "local_hair_support": support,
            "source_hair_mask": source_hair,
            "source_skin_mask": skin,
            **{key: value for key, value in strand_aux.items() if key not in {"anchor_l", "local_hair_support"}},
        }
        return out if return_aux else dense


__all__ = ["HairCoreConfidenceV843", "distance_prior_v843"]
