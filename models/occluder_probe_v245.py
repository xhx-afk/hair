"""Explicit foreground occluder ownership for V2.45 diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v245_death_test_common import composite
from models.v830_runtime_inputs import PARSER_LABELS_V830


class OccluderProbeV245:
    @staticmethod
    def soften(mask: torch.Tensor) -> torch.Tensor:
        """Keep the foreground edge to a one-pixel soft transition."""
        return F.avg_pool2d(mask.float().clamp(0, 1), 3, stride=1, padding=1).clamp(0, 1)

    def real_mask(self, parser_labels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        labels = parser_labels.to(reference.device)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        if labels.shape[-2:] != reference.shape[-2:]:
            labels = F.interpolate(labels.float(), size=reference.shape[-2:], mode="nearest").long()
        selected = (PARSER_LABELS_V830["eye_glasses"], PARSER_LABELS_V830["hat"], PARSER_LABELS_V830["ear_ring"])
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in selected:
            mask |= labels == label
        return mask.float()

    def synthetic_masks(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, height, width = reference.shape
        device = reference.device
        yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        x = xx.float() / max(width - 1, 1)
        y = yy.float() / max(height - 1, 1)
        bar = ((y - 0.43).abs() < 0.012).float()
        arc = (((x - 0.50).pow(2) + (y - 0.48).pow(2)).sqrt() - 0.28).abs().lt(0.014).float()
        line = ((x - (0.16 + 0.68 * y)).abs() < 0.004).float()
        masks = torch.stack((bar, arc, line), dim=0).view(1, 3, height, width)
        masks = masks.expand(reference.size(0), -1, -1, -1)
        return {"horizontal_bar": masks[:, 0:1], "headphone_arc": masks[:, 1:2], "earphone_line": masks[:, 2:3]}

    @staticmethod
    def apply(*, base_rgb: torch.Tensor, hair_rgb: torch.Tensor, hair_alpha: torch.Tensor,
              occluder_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hair_composite = composite(base_rgb, hair_rgb, hair_alpha)
        protected = (occluder_alpha * base_rgb + (1.0 - occluder_alpha) * hair_composite).clamp(0, 1)
        current = hair_composite
        return current, protected


__all__ = ["OccluderProbeV245"]
