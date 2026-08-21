"""Distribution-based intrinsic-tone metrics for V2.39."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import rgb_to_lab
from utils.v238_metrics import v238_metric_tensors


def _masked_values(value: torch.Tensor, mask: torch.Tensor, batch: int) -> torch.Tensor:
    pixels = value[batch].flatten()[mask[batch].flatten() > 0.5]
    return pixels if pixels.numel() else value.new_zeros((1,))


def _median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.stack([_masked_values(value, mask, i).median() for i in range(value.size(0))])


def _quantile_error(reference: torch.Tensor, final: torch.Tensor, mask_ref: torch.Tensor, mask_final: torch.Tensor, q: float) -> torch.Tensor:
    return torch.stack([
        (_masked_values(reference, mask_ref, i).quantile(q) - _masked_values(final, mask_final, i).quantile(q)).abs()
        for i in range(reference.size(0))
    ])


def _subset_or_fallback(mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    usable = mask.flatten(1).sum(1, keepdim=True) > 0
    return torch.where(usable.view(-1, 1, 1, 1), mask, fallback)


def v239_metric_tensors(*, base_rgb: torch.Tensor, final_rgb: torch.Tensor,
                        color_reference_rgb: torch.Tensor, coarse_target_hair_mask: torch.Tensor,
                        source_face_mask: torch.Tensor, hair_alpha_final: torch.Tensor,
                        allowed_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics = v238_metric_tensors(
        base_rgb=base_rgb, final_rgb=final_rgb, color_reference_rgb=color_reference_rgb,
        coarse_target_hair_mask=coarse_target_hair_mask, source_face_mask=source_face_mask,
        hair_alpha_final=hair_alpha_final, allowed_hair_mask=allowed_hair_mask,
    )
    ref_lab = rgb_to_lab(color_reference_rgb)
    final_lab = rgb_to_lab(final_rgb)
    ref_mask = reference_hair_mask.float().clamp(0, 1)
    final_mask = (hair_alpha_final.float().clamp(0, 1) >= 0.5).float()
    ref_l, final_l = ref_lab[:, :1], final_lab[:, :1]
    ref_ab, final_ab = ref_lab[:, 1:], final_lab[:, 1:]
    ref_chroma = torch.linalg.vector_norm(ref_ab, dim=1, keepdim=True)
    final_chroma = torch.linalg.vector_norm(final_ab, dim=1, keepdim=True)
    ref_hue = torch.atan2(ref_ab[:, 1:2], ref_ab[:, 0:1])
    final_hue = torch.atan2(final_ab[:, 1:2], final_ab[:, 0:1])
    hue_delta = torch.atan2(torch.sin(final_hue - ref_hue), torch.cos(final_hue - ref_hue)).abs() * (180.0 / torch.pi)
    low_ref = ref_mask * (ref_chroma < 12.0).float()
    normal_ref = ref_mask * (ref_chroma >= 12.0).float() * (ref_chroma <= 28.0).float()
    high_ref = ref_mask * (ref_chroma > 28.0).float()
    low_ref = _subset_or_fallback(low_ref, ref_mask)
    normal_ref = _subset_or_fallback(normal_ref, ref_mask)
    high_ref = _subset_or_fallback(high_ref, ref_mask)
    # A geometry-independent distribution comparison: each side contributes
    # robust quantiles, avoiding false pixel-wise penalties for different haircuts.
    metrics.update({
        "hair_median_l_error": (_median(ref_l, ref_mask) - _median(final_l, final_mask)).abs(),
        "hair_median_ab_error": (_median(ref_ab.norm(dim=1, keepdim=True), ref_mask) - _median(final_ab.norm(dim=1, keepdim=True), final_mask)).abs(),
        "hair_l_q10_error": _quantile_error(ref_l, final_l, ref_mask, final_mask, 0.10),
        "hair_l_q25_error": _quantile_error(ref_l, final_l, ref_mask, final_mask, 0.25),
        "hair_l_q50_error": _quantile_error(ref_l, final_l, ref_mask, final_mask, 0.50),
        "hair_l_q75_error": _quantile_error(ref_l, final_l, ref_mask, final_mask, 0.75),
        "hair_l_q90_error": _quantile_error(ref_l, final_l, ref_mask, final_mask, 0.90),
        "hair_hue_error_deg": _median(hue_delta, ref_mask),
        "hair_chroma_magnitude_error": (_median(ref_chroma, ref_mask) - _median(final_chroma, final_mask)).abs(),
        "low_chroma_hair_l_error": (_median(ref_l, low_ref) - _median(final_l, final_mask)).abs(),
        "low_chroma_reference_progress": (_median((final_l - ref_l).abs(), ref_mask) < 7.0).float(),
        "low_chroma_tone_fidelity": torch.exp(-((_median(ref_l, low_ref) - _median(final_l, final_mask)).abs() / 7.0)),
        "normal_chroma_hair_l_error": (_median(ref_l, normal_ref) - _median(final_l, final_mask)).abs(),
        "high_chroma_hair_l_error": (_median(ref_l, high_ref) - _median(final_l, final_mask)).abs(),
    })
    return metrics


__all__ = ["v239_metric_tensors"]
