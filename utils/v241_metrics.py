"""V2.41 stable-hue, gamut, and full-overlap metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from utils.v240_metrics import v240_metric_tensors
from statistics import median


def _masked_stats(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = value.flatten(1)
    mask = mask.flatten(1).float()
    denom = mask.sum(1).clamp_min(1e-6)
    mean = (value * mask).sum(1) / denom
    q90_values = []
    for index in range(value.size(0)):
        pixels = value[index][mask[index] > 0]
        q90_values.append(torch.quantile(pixels, 0.90) if pixels.numel() else value.new_tensor(0.0))
    q90 = torch.stack(q90_values)
    maximum = (value * mask).amax(1)
    return torch.nan_to_num(mean), torch.nan_to_num(q90), torch.nan_to_num(maximum)


def _angle_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs() * (180.0 / torch.pi)


def _masked_circular_mean(angle: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.flatten(1).float()
    angle = angle.flatten(1)
    sin_mean = (torch.sin(angle) * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
    cos_mean = (torch.cos(angle) * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
    valid = mask.sum(1) > 0
    result = torch.atan2(sin_mean, cos_mean)
    return torch.where(valid, result, torch.zeros_like(result))


def _masked_quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    output = []
    for batch in range(value.size(0)):
        pixels = value[batch, 0][mask[batch, 0] > 0.5]
        output.append(torch.quantile(pixels, q) if pixels.numel() else value.new_tensor(50.0))
    return torch.stack(output).view(-1, 1, 1, 1)


def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = []
    for batch in range(value.size(0)):
        pixels = value[batch].flatten()[mask[batch].flatten() > 0.5]
        output.append(pixels.median() if pixels.numel() else value.new_tensor(0.0))
    return torch.stack(output)


def v241_metric_tensors(*, base_rgb: torch.Tensor, final_rgb: torch.Tensor,
                        color_reference_rgb: torch.Tensor, coarse_target_hair_mask: torch.Tensor,
                        source_face_mask: torch.Tensor, source_skin_mask: torch.Tensor,
                        hair_alpha_final: torch.Tensor, allowed_hair_mask: torch.Tensor,
                        reference_hair_mask: torch.Tensor, strong_anchor_rgb: torch.Tensor,
                        face_contact_ring: torch.Tensor, face_intrusion_risk: torch.Tensor,
                        anchor_hair_evidence: torch.Tensor | None = None,
                        stable_hue_unit_ab: torch.Tensor | None = None,
                        hue_metric_valid: torch.Tensor | None = None,
                        gamut_aux: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    metrics = v240_metric_tensors(
        base_rgb=base_rgb, final_rgb=final_rgb, color_reference_rgb=color_reference_rgb,
        coarse_target_hair_mask=coarse_target_hair_mask, source_face_mask=source_face_mask,
        hair_alpha_final=hair_alpha_final, allowed_hair_mask=allowed_hair_mask,
        reference_hair_mask=reference_hair_mask, strong_anchor_rgb=strong_anchor_rgb,
        face_contact_ring=face_contact_ring, face_intrusion_risk=face_intrusion_risk,
    )
    ref_lab, final_lab = rgb_to_lab(color_reference_rgb), rgb_to_lab(final_rgb)
    ref_mask = reference_hair_mask.float().clamp(0, 1)
    final_mask = (hair_alpha_final.float().clamp(0, 1) >= 0.5).float()
    stable_mask = (hair_alpha_final >= 0.85).float()
    ref_ab = ref_lab[:, 1:]
    final_ab = final_lab[:, 1:]
    ref_c = torch.linalg.vector_norm(ref_ab, dim=1, keepdim=True)
    final_c = torch.linalg.vector_norm(final_ab, dim=1, keepdim=True)
    # Compare output against the robust single reference direction, rather
    # than allowing per-pixel reference highlights to hide hue switching.
    if stable_hue_unit_ab is None:
        ref_valid = reference_hair_mask.float().clamp(0, 1)
        ref_unit = ref_ab / ref_c.clamp_min(1e-4)
        ref_weight = ref_c.clamp(0, 30) * ref_valid
        ref_dir = (ref_unit * ref_weight).flatten(2).sum(2) / ref_weight.flatten(2).sum(2).clamp_min(1e-4)
        ref_dir = ref_dir / torch.linalg.vector_norm(ref_dir, dim=1, keepdim=True).clamp_min(1e-4)
    else:
        ref_dir = stable_hue_unit_ab.flatten(2).mean(2)
    ref_theta = torch.atan2(ref_dir[:, 1:2], ref_dir[:, 0:1]).view(-1, 1, 1, 1)
    ref_theta = ref_theta.expand(-1, -1, final_ab.size(-2), final_ab.size(-1))
    final_theta = torch.atan2(final_ab[:, 1:2], final_ab[:, 0:1])
    reference_hue_valid = torch.ones_like(stable_mask[:, :1, :1, :1]) if hue_metric_valid is None else hue_metric_valid.float().clamp(0, 1)
    final_hue_pixel_count = (stable_mask * (final_c >= 5.0).float()).flatten(1).sum(1)
    final_hue_support_valid = (final_hue_pixel_count >= 32).float().view(-1, 1, 1, 1)
    effective_hue_valid = reference_hue_valid * final_hue_support_valid
    valid = stable_mask * (final_c >= 5.0).float() * effective_hue_valid
    delta = _angle_delta(final_theta, ref_theta)
    high_valid = valid * (final_c > 15.0).float()
    high_chroma_hue_pixel_count = high_valid.flatten(1).sum(1)
    high_chroma_hue_metric_valid = effective_hue_valid * (high_chroma_hue_pixel_count >= 32).float().view(-1, 1, 1, 1)
    final_l = final_lab[:, :1]
    q80 = _masked_quantile(final_l, stable_mask, 0.80)
    highlight = valid * (final_l >= q80).float()
    mid = valid * (final_l < q80).float()
    highlight_metric_valid = (highlight.flatten(1).sum(1) >= 32).float() * effective_hue_valid.flatten(1).mean(1)
    mid_metric_valid = (mid.flatten(1).sum(1) >= 32).float() * effective_hue_valid.flatten(1).mean(1)
    highlight_drift_valid = highlight_metric_valid * mid_metric_valid
    delta_mean, _, _ = _masked_stats(delta, valid)
    spatial_std = torch.sqrt((_masked_stats(delta.pow(2), valid)[0] - delta_mean.pow(2)).clamp_min(0))
    outlier_fraction = _masked_stats((delta > 20.0).float(), valid)[0]
    high_outlier_fraction = _masked_stats((delta > 20.0).float(), high_valid)[0]
    highlight_error = _masked_stats(delta, highlight)[0]
    mid_error = _masked_stats(delta, mid)[0]
    highlight_angle = _masked_circular_mean(final_theta, highlight)
    mid_angle = _masked_circular_mean(final_theta, mid)
    drift = _angle_delta(highlight_angle, mid_angle)
    metrics.update({
        "hair_hue_spatial_std": spatial_std,
        "hair_hue_outlier_fraction": outlier_fraction,
        "high_chroma_hue_outlier_fraction": high_outlier_fraction,
        "highlight_hue_error_deg": highlight_error,
        "mid_hue_error_deg": mid_error,
        "highlight_hue_drift": drift,
        "stable_hue_error_deg": delta_mean,
        "reference_hue_metric_valid": reference_hue_valid.flatten(1).mean(1),
        "final_hue_support_valid": final_hue_support_valid.flatten(1).mean(1),
        "hue_metric_valid": effective_hue_valid.flatten(1).mean(1),
        "effective_hue_metric_valid": effective_hue_valid.flatten(1).mean(1),
        "final_hue_pixel_count": final_hue_pixel_count,
        "high_chroma_hue_pixel_count": high_chroma_hue_pixel_count,
        "high_chroma_hue_metric_valid": high_chroma_hue_metric_valid.flatten(1).mean(1),
        "highlight_hue_metric_valid": highlight_metric_valid,
        "mid_hue_metric_valid": mid_metric_valid,
        "highlight_hue_drift_valid": highlight_drift_valid,
    })
    if "hair_hue_error_deg" in metrics:
        metrics["legacy_xy_hue_error_deg"] = metrics.pop("hair_hue_error_deg")
    coarse = coarse_target_hair_mask.float().clamp(0, 1)
    face_overlap = coarse * source_face_mask.float().clamp(0, 1)
    skin_overlap = coarse * source_skin_mask.float().clamp(0, 1)
    deep_skin = skin_overlap * (1.0 - face_contact_ring.float().clamp(0, 1))
    evidence = torch.ones_like(skin_overlap) if anchor_hair_evidence is None else anchor_hair_evidence.float().clamp(0, 1)
    uncertain_skin = skin_overlap * (1.0 - evidence)
    deep_uncertain_skin = deep_skin * (evidence < 0.65).float()
    bleed = (final_rgb - base_rgb).abs().mean(1, keepdim=True)
    for prefix, zone in (("face_overlap", face_overlap), ("skin_overlap", skin_overlap),
                         ("deep_skin_overlap", deep_skin), ("uncertain_skin_overlap", uncertain_skin),
                         ("deep_uncertain_skin_overlap", deep_uncertain_skin)):
        mean, p90, maximum = _masked_stats(bleed, zone)
        metrics[f"{prefix}_bleed"] = mean
        metrics[f"{prefix}_bleed_p90"] = p90
        metrics[f"{prefix}_bleed_max"] = maximum
    if gamut_aux is not None:
        for name in ("gamut_unsafe_fraction", "gamut_compressed_fraction", "gamut_heavy_compression_fraction", "median_gamut_scale", "p10_gamut_scale"):
            value = gamut_aux.get(name)
            if value is not None:
                metrics[name] = value if value.dim() == 1 else value.flatten(1).mean(1)
    else:
        zero = final_rgb.new_zeros(final_rgb.size(0))
        metrics.update({name: zero for name in ("gamut_unsafe_fraction", "gamut_compressed_fraction", "gamut_heavy_compression_fraction", "median_gamut_scale", "p10_gamut_scale")})
    visible_face = source_face_mask.float().clamp(0, 1) * (1.0 - hair_alpha_final.float().clamp(0, 1))
    visible_skin = source_skin_mask.float().clamp(0, 1) * (1.0 - hair_alpha_final.float().clamp(0, 1))
    visible_background = (1.0 - source_face_mask.float().clamp(0, 1) - source_skin_mask.float().clamp(0, 1)).clamp(0, 1)
    visible_background = visible_background * (1.0 - hair_alpha_final.float().clamp(0, 1))
    for prefix, zone in (("visible_face", visible_face), ("visible_skin", visible_skin),
                         ("visible_background", visible_background)):
        mean, p90, maximum = _masked_stats(bleed, zone)
        metrics[f"{prefix}_rgb_change_from_base"] = mean
        metrics[f"{prefix}_rgb_change_p90"] = p90
        metrics[f"{prefix}_rgb_change_max"] = maximum
    # These names are retained only as diagnostics. V2.41.2 gates use the
    # distribution-based metrics above, never coordinate-wise XY comparisons.
    legacy_names = {
        "hair_reference_progress": "legacy_xy_reference_progress",
        "base_leakage_fraction": "legacy_xy_base_leakage_fraction",
        "hair_low_transfer_fraction": "legacy_xy_low_transfer_fraction",
        "hair_chroma_ab_error": "legacy_xy_chroma_ab_error",
    }
    for old, legacy in legacy_names.items():
        if old in metrics:
            metrics[legacy] = metrics[old]
    ref_median_l = _masked_median(ref_lab[:, :1], ref_mask)
    final_median_l = _masked_median(final_lab[:, :1], final_mask)
    ref_global_chroma = _masked_median(ref_c, ref_mask)
    low_chroma_valid = (ref_global_chroma < 12.0).float()
    low_chroma_l_error = (ref_median_l - final_median_l).abs()
    metrics.update({
        "reference_hair_median_l": ref_median_l,
        "final_hair_median_l": final_median_l,
        "reference_global_chroma": ref_global_chroma,
        "low_chroma_sample_valid": low_chroma_valid,
        "low_chroma_hair_l_error": low_chroma_l_error,
        "low_chroma_tone_fidelity": torch.exp(-low_chroma_l_error / 7.0),
    })
    face_boundary_ring = (F.max_pool2d(source_face_mask.float(), 7, stride=1, padding=3) -
                          (-F.max_pool2d(-source_face_mask.float(), 7, stride=1, padding=3))).clamp(0, 1)
    alpha_local_range = F.max_pool2d(hair_alpha_final.float(), 7, stride=1, padding=3) - (-F.max_pool2d(-hair_alpha_final.float(), 7, stride=1, padding=3))
    boundary_zone = face_boundary_ring * coarse
    metrics["face_boundary_alpha_jump"] = _masked_stats(alpha_local_range, boundary_zone)[0]
    return {key: torch.nan_to_num(value) for key, value in metrics.items()}


def aggregate_v2412_records(records: list[dict], *, min_fraction: float = 0.10,
                            min_samples: int = 5) -> tuple[dict[str, float], list[str]]:
    """Aggregate V2.41.2 records without treating invalid hue as zero error."""
    if not records:
        return {"count": 0}, ["V2412_NO_VALID_SAMPLES"]
    keys = sorted({key for row in records for key in row if key != "sample_id"})
    summary = {
        f"median_{key}": float(median([float(row.get(key, 0.0)) for row in records]))
        for key in keys
    }
    hue_rows = [row for row in records if float(row.get("effective_hue_metric_valid", row.get("hue_metric_valid", 0.0))) > 0.5]
    high_rows = [row for row in records if float(row.get("high_chroma_hue_metric_valid", 0.0)) > 0.5]
    highlight_rows = [row for row in records if float(row.get("highlight_hue_drift_valid", 0.0)) > 0.5]
    low_rows = [row for row in records if float(row.get("low_chroma_sample_valid", 0.0)) > 0.5]
    for key in ("stable_hue_error_deg", "hair_hue_outlier_fraction"):
        if hue_rows:
            summary[f"median_{key}"] = float(median([float(row.get(key, 0.0)) for row in hue_rows]))
    if high_rows:
        summary["median_high_chroma_hue_outlier_fraction"] = float(median([float(row.get("high_chroma_hue_outlier_fraction", 0.0)) for row in high_rows]))
    if highlight_rows:
        summary["median_highlight_hue_drift"] = float(median([float(row.get("highlight_hue_drift", 0.0)) for row in highlight_rows]))
    if low_rows:
        for key in ("low_chroma_hair_l_error", "low_chroma_tone_fidelity"):
            summary[f"median_{key}"] = float(median([float(row.get(key, 0.0)) for row in low_rows]))
    summary["reference_hue_valid_sample_count"] = sum(float(row.get("reference_hue_metric_valid", 0.0)) > 0.5 for row in records)
    summary["hue_valid_sample_count"] = len(hue_rows)
    summary["high_chroma_hue_valid_sample_count"] = len(high_rows)
    summary["highlight_hue_valid_sample_count"] = len(highlight_rows)
    summary["mid_hue_valid_sample_count"] = sum(float(row.get("mid_hue_metric_valid", 0.0)) > 0.5 for row in records)
    summary["highlight_hue_drift_valid_sample_count"] = len(highlight_rows)
    summary["low_chroma_valid_sample_count"] = len(low_rows)
    summary["count"] = len(records)
    required = max(int(min_samples), int((min_fraction * len(records)) + 0.999999))
    failures = []
    if len(hue_rows) < required:
        failures.append("V2412_HUE_INSUFFICIENT_VALID_SAMPLES")
    if len(high_rows) < required:
        failures.append("V2412_HIGH_CHROMA_HUE_INSUFFICIENT_VALID_SAMPLES")
    if len(highlight_rows) < required:
        failures.append("V2412_HIGHLIGHT_HUE_INSUFFICIENT_VALID_SAMPLES")
    low_required = max(int(min_samples), int((0.05 * len(records)) + 0.999999))
    if len(low_rows) < low_required:
        failures.append("V2412_LOW_CHROMA_INSUFFICIENT_VALID_SAMPLES")
    return summary, failures


__all__ = ["v241_metric_tensors", "aggregate_v2412_records"]
