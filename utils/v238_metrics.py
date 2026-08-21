"""V2.38 ownership, matte, coverage, and corrected chroma metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def v238_metric_tensors(
    *, base_rgb: torch.Tensor, final_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
    coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
    hair_alpha_final: torch.Tensor, allowed_hair_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    base_lab, final_lab, ref_lab = rgb_to_lab(base_rgb), rgb_to_lab(final_rgb), rgb_to_lab(color_reference_rgb)
    hair = coarse_target_hair_mask.float().clamp(0, 1)
    alpha = hair_alpha_final.float().clamp(0, 1)
    allowed = allowed_hair_mask.float().clamp(0, 1)
    face = source_face_mask.float().clamp(0, 1)
    non_hair = (1 - alpha).clamp(0, 1)
    final_ref_error = torch.linalg.vector_norm(final_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
    base_ref_error = torch.linalg.vector_norm(base_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
    final_base_error = torch.linalg.vector_norm(final_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True)
    strength = (final_base_error / base_ref_error.clamp_min(1e-4)).clamp(0, 1)
    undertransfer = (final_ref_error >= 0.7 * base_ref_error).float()
    boundary = (alpha * (1 - F.avg_pool2d(alpha, 7, stride=1, padding=3))).clamp(0, 1)
    stair = (alpha - F.avg_pool2d(alpha, 3, stride=1, padding=1)).abs()
    return {
        "face_rgb_change_from_base": _mean((final_rgb - base_rgb).abs(), face),
        "background_rgb_change_from_base": _mean((final_rgb - base_rgb).abs(), non_hair),
        "hair_chroma_ab_error": _mean(final_ref_error, hair),
        "hair_reference_progress": _mean((final_ref_error < base_ref_error).float(), hair),
        "base_leakage_fraction": _mean((final_base_error < 0.35 * base_ref_error).float(), hair),
        "hair_core_full_transfer_fraction": _mean((strength >= 0.85).float(), alpha >= 0.85),
        "hair_low_transfer_fraction": _mean((strength < 0.50).float(), hair),
        "hair_undertransfer_fraction": _mean(undertransfer, alpha >= 0.85),
        "hair_on_face_coverage_fraction": _mean(alpha, hair * face),
        "blocked_target_hair_fraction": _mean((1 - allowed).clamp(0, 1), hair),
        "boundary_stair_step_score": _mean(stair, boundary),
        "boundary_misalignment_score": _mean((alpha - allowed).abs(), boundary),
        "boundary_artifact_score": _mean((alpha - gaussian_blur_v829(alpha, 2)).abs(), boundary),
    }


__all__ = ["v238_metric_tensors"]
