"""V2.44 carrier-preservation and residual photometric metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v240_metrics import _masked_corr, _masked_mean, _sobel, v240_metric_tensors
from utils.v243_metrics import v243_metric_tensors


def _texture_stats(rgb: torch.Tensor, mask: torch.Tensor, anchor_l: torch.Tensor) -> dict[str, torch.Tensor]:
    lab_l = rgb_to_lab(rgb)[:, :1]
    mid = gaussian_blur_v829(lab_l, 3) - gaussian_blur_v829(lab_l, 11)
    hf = lab_l - gaussian_blur_v829(lab_l, 3)
    anchor_mid = gaussian_blur_v829(anchor_l, 3) - gaussian_blur_v829(anchor_l, 11)
    anchor_hf = anchor_l - gaussian_blur_v829(anchor_l, 3)
    gradient = _sobel(lab_l)
    anchor_gradient = _sobel(anchor_l)
    return {
        "mid_energy": _masked_mean(mid.abs(), mask),
        "hf_energy": _masked_mean(hf.abs(), mask),
        "mid_structure": _masked_corr(mid, anchor_mid, mask),
        "gradient_structure": _masked_corr(gradient, anchor_gradient, mask),
    }


def v244_metric_tensors(*, base_rgb: torch.Tensor, final_rgb: torch.Tensor,
                        carrier_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                        coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
                        source_skin_mask: torch.Tensor, hair_alpha_final: torch.Tensor,
                        allowed_hair_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
                        face_contact_ring: torch.Tensor, face_intrusion_risk: torch.Tensor,
                        independent_flyaway_candidate: torch.Tensor | None = None,
                        independent_flyaway_recovery: torch.Tensor | None = None,
                        scene_illumination_ab: torch.Tensor | None = None,
                        total_gamut_scale: torch.Tensor | None = None,
                        pre_gamut_scale: torch.Tensor | None = None,
                        final_gamut_scale: torch.Tensor | None = None,
                        **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics = v240_metric_tensors(
        base_rgb=base_rgb, final_rgb=final_rgb, color_reference_rgb=kwargs.get("color_reference_rgb", carrier_rgb),
        coarse_target_hair_mask=coarse_target_hair_mask, source_face_mask=source_face_mask,
        hair_alpha_final=hair_alpha_final, allowed_hair_mask=allowed_hair_mask,
        reference_hair_mask=reference_hair_mask, strong_anchor_rgb=strong_anchor_rgb,
        face_contact_ring=face_contact_ring, face_intrusion_risk=face_intrusion_risk,
    )
    core = (hair_alpha_final >= 0.85).float() * coarse_target_hair_mask.float().clamp(0, 1)
    anchor_l = rgb_to_lab(strong_anchor_rgb)[:, :1]
    carrier_stats = _texture_stats(carrier_rgb, core, anchor_l)
    final_stats = _texture_stats(final_rgb, core, anchor_l)
    carrier_l = rgb_to_lab(carrier_rgb)[:, :1]
    final_l = rgb_to_lab(final_rgb)[:, :1]
    final_mid = gaussian_blur_v829(final_l, 3) - gaussian_blur_v829(final_l, 11)
    carrier_mid = gaussian_blur_v829(carrier_l, 3) - gaussian_blur_v829(carrier_l, 11)
    final_hf = final_l - gaussian_blur_v829(final_l, 3)
    carrier_hf = carrier_l - gaussian_blur_v829(carrier_l, 3)
    metrics.update({
        "v242_mid_energy": carrier_stats["mid_energy"],
        "v244_mid_energy": final_stats["mid_energy"],
        "v242_hf_energy": carrier_stats["hf_energy"],
        "v244_hf_energy": final_stats["hf_energy"],
        "v242_mid_structure": carrier_stats["mid_structure"],
        "v244_mid_structure_corr": _masked_corr(final_mid, carrier_mid, core),
        "v242_gradient_structure": carrier_stats["gradient_structure"],
        "v244_gradient_structure_corr": _masked_corr(_sobel(final_l), _sobel(carrier_l), core),
        "mid_energy_ratio": final_stats["mid_energy"] / _masked_mean(anchor_l.abs(), core).clamp_min(1e-4),
        "hf_energy_ratio": final_stats["hf_energy"] / _masked_mean(anchor_l.abs(), core).clamp_min(1e-4),
        "v242_relative_mid_energy": final_stats["mid_energy"] / carrier_stats["mid_energy"].clamp_min(1e-4),
        "v242_relative_hf_energy": final_stats["hf_energy"] / carrier_stats["hf_energy"].clamp_min(1e-4),
        "l_detail_loss": _masked_mean((final_mid - carrier_mid).abs(), core),
        "ab_detail_loss": _masked_mean((rgb_to_lab(final_rgb)[:, 1:] - rgb_to_lab(carrier_rgb)[:, 1:]).norm(dim=1, keepdim=True), core),
        "new_hair_coverage": _masked_mean(hair_alpha_final, coarse_target_hair_mask),
        "new_hair_on_face_coverage": _masked_mean(hair_alpha_final, coarse_target_hair_mask * source_face_mask.float().clamp(0, 1)),
    })
    # The V2.44 gate names are carrier-relative, while V2.40 metrics retain
    # their historical strong-anchor comparison fields.
    metrics["mid_structure_corr"] = metrics["v244_mid_structure_corr"]
    metrics["gradient_structure_corr"] = metrics["v244_gradient_structure_corr"]
    safe_dense_core = kwargs.get("safe_dense_core")
    uncertain_core = kwargs.get("uncertain_core")
    anchor_hair_evidence = kwargs.get("anchor_hair_evidence")
    strand_structure_confidence = kwargs.get("strand_structure_confidence")
    metrics.update(v243_metric_tensors(
        target_hair_mask=coarse_target_hair_mask,
        coarse_target_hair_mask=coarse_target_hair_mask,
        hair_alpha_final=hair_alpha_final,
        safe_dense_core=core if safe_dense_core is None else safe_dense_core,
        uncertain_core=torch.zeros_like(core) if uncertain_core is None else uncertain_core,
        source_skin_mask=source_skin_mask,
        anchor_hair_evidence=torch.ones_like(core) if anchor_hair_evidence is None else anchor_hair_evidence,
        strand_structure_confidence=torch.ones_like(core) if strand_structure_confidence is None else strand_structure_confidence,
        hair_support=kwargs.get("hair_support"),
        flyaway_candidate=independent_flyaway_candidate,
        final_l=final_l,
        reference_chroma=rgb_to_lab(final_rgb)[:, 1:].norm(dim=1, keepdim=True),
        scene_illumination_ab=scene_illumination_ab,
    ))
    if total_gamut_scale is None:
        if pre_gamut_scale is not None and final_gamut_scale is not None:
            total_gamut_scale = (pre_gamut_scale * final_gamut_scale).clamp(0, 1)
        else:
            total_gamut_scale = final_rgb.new_ones(final_rgb.size(0), 1, final_rgb.size(2), final_rgb.size(3))
    heavy = (total_gamut_scale < 0.50).float()
    compressed = (total_gamut_scale < 0.999).float()
    metrics.update({
        "total_gamut_scale": total_gamut_scale.flatten(1).mean(1),
        "total_gamut_compressed_fraction": compressed.flatten(1).mean(1),
        "total_gamut_heavy_compression_fraction": heavy.flatten(1).mean(1),
        "p10_total_gamut_scale": torch.quantile(total_gamut_scale.flatten(1), 0.10, dim=1),
    })
    candidate = torch.zeros_like(hair_alpha_final) if independent_flyaway_candidate is None else independent_flyaway_candidate.float().clamp(0, 1)
    recovery = ((hair_alpha_final > 0.1).float() * candidate if independent_flyaway_recovery is None else independent_flyaway_recovery.float()).clamp(0, 1)
    metrics["independent_flyaway_recovery"] = _masked_mean(recovery, candidate)
    if scene_illumination_ab is not None:
        metrics["scene_illumination_shift_deg"] = (
            torch.atan2(scene_illumination_ab[:, 1:2], scene_illumination_ab[:, 0:1])
            * (180.0 / torch.pi)
        ).flatten(1).mean(1).abs()
    metrics["low_confidence_skin_hair"] = _masked_mean(
        (hair_alpha_final > 0.5).float(),
        source_skin_mask.float() * coarse_target_hair_mask.float() * (hair_alpha_final < 0.55).float(),
    )
    return {key: torch.nan_to_num(value) for key, value in metrics.items()}


__all__ = ["v244_metric_tensors"]
