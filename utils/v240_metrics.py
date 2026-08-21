"""Texture preservation and face-contact bleed metrics for V2.40."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v239_metrics import v239_metric_tensors


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def _masked_corr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(x)
    denom = mask.flatten(1).sum(1).clamp_min(2.0)
    mx = (x * mask).flatten(1).sum(1) / denom
    my = (y * mask).flatten(1).sum(1) / denom
    dx, dy = (x - mx[:, None, None, None]) * mask, (y - my[:, None, None, None]) * mask
    return (dx * dy).flatten(1).sum(1) / (torch.sqrt(dx.pow(2).flatten(1).sum(1) * dy.pow(2).flatten(1).sum(1)).clamp_min(1e-6))


def _sobel(value: torch.Tensor) -> torch.Tensor:
    kx = value.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = value.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    return torch.sqrt(F.conv2d(value, kx, padding=1).pow(2) + F.conv2d(value, ky, padding=1).pow(2) + 1e-6)


def v240_metric_tensors(*, base_rgb: torch.Tensor, final_rgb: torch.Tensor,
                        color_reference_rgb: torch.Tensor, coarse_target_hair_mask: torch.Tensor,
                        source_face_mask: torch.Tensor, hair_alpha_final: torch.Tensor,
                        allowed_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                        strong_anchor_rgb: torch.Tensor, face_contact_ring: torch.Tensor,
                        face_intrusion_risk: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics = v239_metric_tensors(
        base_rgb=base_rgb, final_rgb=final_rgb, color_reference_rgb=color_reference_rgb,
        coarse_target_hair_mask=coarse_target_hair_mask, source_face_mask=source_face_mask,
        hair_alpha_final=hair_alpha_final, allowed_hair_mask=allowed_hair_mask,
        reference_hair_mask=reference_hair_mask,
    )
    anchor_l = rgb_to_lab(strong_anchor_rgb)[:, :1]
    final_l = rgb_to_lab(final_rgb)[:, :1]
    mid_anchor = gaussian_blur_v829(anchor_l, 3) - gaussian_blur_v829(anchor_l, 11)
    mid_final = gaussian_blur_v829(final_l, 3) - gaussian_blur_v829(final_l, 11)
    hf_anchor = anchor_l - gaussian_blur_v829(anchor_l, 3)
    hf_final = final_l - gaussian_blur_v829(final_l, 3)
    core = (hair_alpha_final >= 0.85).float()
    contact_zone = face_contact_ring.float().clamp(0, 1) * source_face_mask.float().clamp(0, 1)
    non_core_contact = contact_zone * (1.0 - core)
    gradient_anchor, gradient_final = _sobel(anchor_l), _sobel(final_l)
    mid_energy = _masked_mean(mid_final.abs(), core) / _masked_mean(mid_anchor.abs(), core).clamp_min(1e-4)
    hf_energy = _masked_mean(hf_final.abs(), core) / _masked_mean(hf_anchor.abs(), core).clamp_min(1e-4)
    metrics.update({
        "mid_energy_ratio": mid_energy,
        "hf_energy_ratio": hf_energy,
        "mid_structure_corr": _masked_corr(mid_final, mid_anchor, core),
        "gradient_structure_corr": _masked_corr(gradient_final, gradient_anchor, core),
        "face_contact_bleed": _masked_mean((final_rgb - base_rgb).abs().mean(1, keepdim=True), non_core_contact),
        "face_contact_bleed_p90": torch.quantile(((final_rgb - base_rgb).abs().mean(1, keepdim=True) * non_core_contact).flatten(1), 0.90, dim=1),
        "face_contact_bleed_max": ((final_rgb - base_rgb).abs().mean(1, keepdim=True) * non_core_contact).flatten(1).amax(1),
        "face_contact_risk_mean": _masked_mean(face_intrusion_risk, contact_zone),
    })
    return metrics


__all__ = ["v240_metric_tensors"]
