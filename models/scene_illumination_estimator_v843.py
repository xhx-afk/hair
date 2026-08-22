"""Estimate a restrained chromatic scene-light tint from the target context."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab


class SceneIlluminationEstimatorV843:
    def __init__(self, *, outer_radius: int = 15, inner_radius: int = 5,
                 scene_mix: float = 0.15) -> None:
        self.outer_radius, self.inner_radius = int(outer_radius), int(inner_radius)
        self.scene_mix = min(max(float(scene_mix), 0.0), 0.15)

    @staticmethod
    def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        radius = max(int(radius), 1)
        return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)

    def __call__(self, *, target_rgb: torch.Tensor, target_hair_mask: torch.Tensor,
                 return_aux: bool = False, **_: torch.Tensor):
        mask = target_hair_mask.float().clamp(0, 1)
        outer = self._dilate(mask, self.outer_radius)
        inner = self._dilate(mask, self.inner_radius)
        ring = (outer - inner).clamp(0, 1)
        lab = rgb_to_lab(target_rgb)
        ab = lab[:, 1:]
        weights = ring * (ab.square().sum(1, keepdim=True).sqrt() < 75.0).float()
        denom = weights.flatten(1).sum(1, keepdim=True).clamp_min(1.0)
        scene_ab = (ab * weights).flatten(2).sum(2).view(-1, 2, 1, 1) / denom.view(-1, 1, 1, 1)
        reliability = (weights.flatten(1).sum(1) / ring.flatten(1).sum(1).clamp_min(1.0)).view(-1, 1, 1, 1)
        scene_ab = scene_ab.clamp(-40.0, 40.0)
        aux = {
            "context_ring": ring,
            "scene_illumination_ab": scene_ab,
            "scene_illumination_reliability": reliability.clamp(0, 1),
            "scene_mix": target_rgb.new_full((target_rgb.size(0), 1, 1, 1), self.scene_mix),
        }
        return aux if return_aux else scene_ab


__all__ = ["SceneIlluminationEstimatorV843"]
