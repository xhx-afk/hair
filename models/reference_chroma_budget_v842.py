"""Reference-aware soft chroma budget before the existing gamut solver."""

from __future__ import annotations

import torch


def apply_reference_chroma_budget_v842(chroma: torch.Tensor, *, lightness: torch.Tensor,
                                       reference_chroma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce only extreme-L chroma, preserving normal hair chroma magnitude."""
    del reference_chroma
    extreme_l = ((lightness - 50.0).abs() / 50.0).clamp(0, 1)
    budget = (1.0 - 0.25 * extreme_l.square()).clamp(0.70, 1.0)
    return chroma * budget, budget


__all__ = ["apply_reference_chroma_budget_v842"]
