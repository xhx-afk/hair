"""Reference-aware chroma budgeting for V2.43."""

from __future__ import annotations

import torch


def apply_reference_chroma_budget_v843(
    chroma: torch.Tensor, *, lightness: torch.Tensor,
    reference_chroma: torch.Tensor, shadow_scale: torch.Tensor | None = None,
    highlight_scale: torch.Tensor | None = None,
    minimum_scale: float = 0.55,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep reference magnitude authoritative, then apply photometric scales."""
    budget = torch.ones_like(chroma)
    reference_ratio = (reference_chroma.float().clamp_min(1e-4) / chroma.float().clamp_min(1e-4)).clamp(0.70, 1.0)
    budget = budget * reference_ratio
    if shadow_scale is not None:
        budget = budget * shadow_scale.float().clamp(0.30, 1.0)
    if highlight_scale is not None:
        budget = budget * highlight_scale.float().clamp(0.40, 1.0)
    # Only a small safety roll-off is applied for extreme L; normal midtones
    # retain their requested chroma instead of receiving a global saturation cut.
    extreme = ((lightness.float() - 50.0).abs() / 50.0).clamp(0, 1)
    budget = (budget * (1.0 - 0.10 * extreme.square())).clamp(float(minimum_scale), 1.0)
    return chroma * budget, budget


__all__ = ["apply_reference_chroma_budget_v843"]
