"""Reference-aware soft chroma budget before the existing gamut solver."""

from __future__ import annotations

import torch


def apply_reference_chroma_budget_v842(chroma: torch.Tensor, *, lightness: torch.Tensor,
                                       reference_chroma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce only extreme-L chroma while retaining the reference magnitude."""
    reference_chroma = reference_chroma.float().clamp_min(1e-4)
    extreme_l = ((lightness - 50.0).abs() / 50.0).clamp(0, 1)
    # A carrier may add detail, but it must not make the result exceed the
    # requested reference chroma at shadow/highlight extremes.
    reference_ratio = (reference_chroma / chroma.float().clamp_min(1e-4)).clamp(0.70, 1.0)
    budget = (reference_ratio * (1.0 - 0.25 * extreme_l.square())).clamp(0.70, 1.0)
    return chroma * budget, budget


__all__ = ["apply_reference_chroma_budget_v842"]
