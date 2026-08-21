"""V2.37 metrics compare final appearance against Base/Source ownership."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def v237_metric_tensors(
    *, base_rgb: torch.Tensor, final_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor, face_mask: torch.Tensor, hair_alpha: torch.Tensor,
) -> dict[str, torch.Tensor]:
    base_lab, final_lab, ref_lab = rgb_to_lab(base_rgb), rgb_to_lab(final_rgb), rgb_to_lab(color_reference_rgb)
    hair = target_hair_mask.float().clamp(0, 1)
    alpha = hair_alpha.float().clamp(0, 1)
    face = face_mask.float().clamp(0, 1)
    non_hair = (1.0 - alpha).clamp(0, 1)
    final_ab_error = torch.linalg.vector_norm(final_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
    base_ab_error = torch.linalg.vector_norm(base_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
    final_base_error = torch.linalg.vector_norm(final_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True)
    progress = (final_ab_error < base_ab_error).float()
    strength = (final_ab_error / base_ab_error.clamp_min(1e-4)).clamp(0, 1)
    sorted_error = final_ab_error.flatten(1).sort(dim=1).values
    p90 = sorted_error[:, int(sorted_error.size(1) * 0.90)].view(-1)
    robust_max = sorted_error[:, int(sorted_error.size(1) * 0.98)].view(-1)
    hf_final = final_rgb - gaussian_blur_v829(final_rgb, 2)
    hf_base = base_rgb - gaussian_blur_v829(base_rgb, 2)
    return {
        "face_rgb_change_from_base": _masked_mean((final_rgb - base_rgb).abs(), face),
        "face_rgb_change_from_base_max": ((final_rgb - base_rgb).abs() * face).amax(dim=(1, 2, 3)),
        "background_rgb_change_from_base": _masked_mean((final_rgb - base_rgb).abs(), non_hair),
        "non_hair_rgb_change_from_base": ((final_rgb - base_rgb).abs() * non_hair).amax(dim=(1, 2, 3)),
        "hair_chroma_ab_error": _masked_mean(final_ab_error, hair),
        "hair_reference_progress": _masked_mean(progress, hair),
        "hair_reference_ab_distance_mean": _masked_mean(final_ab_error, hair),
        "hair_reference_ab_distance_p90": p90,
        "hair_reference_ab_distance_max_robust": robust_max,
        "base_leakage_fraction": _masked_mean((final_base_error < final_ab_error).float(), hair),
        "hair_core_full_transfer_fraction": _masked_mean((strength <= 0.15).float(), alpha >= 0.85),
        "hair_low_transfer_fraction": _masked_mean((strength >= 0.50).float(), hair),
        "texture_preservation_error": _masked_mean((hf_final - hf_base).abs(), hair),
        "boundary_error": _masked_mean((final_lab[:, 1:] - gaussian_blur_v829(final_lab[:, 1:], 2)).abs(), hair * (1 - alpha)),
    }


__all__ = ["v237_metric_tensors"]
