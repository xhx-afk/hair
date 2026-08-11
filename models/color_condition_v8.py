from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, rgb_to_lab


COLOR_DESCRIPTOR_DIM = 20


@dataclass(frozen=True)
class ColorConditionConfigV8:
    chroma_no_edit_threshold: float = 3.0
    chroma_full_edit_threshold: float = 19.0
    lightness_no_edit_threshold: float = 3.0
    lightness_full_edit_threshold: float = 15.0
    max_global_l_shift: float = 20.0
    min_safe_fraction: float = 0.35
    highlight_mad_scale: float = 1.8
    highlight_global_min_margin: float = 3.0
    highlight_local_l_margin: float = 2.5
    highlight_local_c_margin: float = 1.5
    highlight_chroma_ratio: float = 0.82
    highlight_near_clip: float = 0.94
    highlight_clip_chroma: float = 12.0
    highlight_softness_l: float = 2.0
    highlight_softness_c: float = 1.5
    highlight_local_radius: int = 7
    reference_mask_erode: int = 2
    conditional_luma_bins: int = 9


def _as_image4d(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(1) != 3:
        raise ValueError(f"Expected RGB [B,3,H,W], got shape={tuple(image.shape)}")
    return image.float()


def _as_mask4d(mask: torch.Tensor, size: tuple[int, int], batch: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"Expected mask [B,1,H,W], got shape={tuple(mask.shape)}")
    if mask.size(0) == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    if mask.size(0) != batch:
        raise ValueError(f"Mask batch {mask.size(0)} does not match image batch {batch}")
    mask = mask.float().clamp(0, 1)
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask.clamp(0, 1)


def _to_rgb01(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    image = _as_image4d(image)
    normalized = bool((image.detach().amin() < -0.05).item())
    if normalized:
        image = image * 0.5 + 0.5
    return image.clamp(0, 1), normalized


def _from_rgb01(image: torch.Tensor, normalized: bool) -> torch.Tensor:
    image = image.clamp(0, 1)
    return image * 2.0 - 1.0 if normalized else image


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    values = values.flatten(2)
    weights = weights.flatten(2)
    if weights.size(1) == 1 and values.size(1) != 1:
        weights = weights.expand(-1, values.size(1), -1)

    sorted_values, indices = values.sort(dim=-1)
    sorted_weights = weights.gather(-1, indices)
    cumulative = sorted_weights.cumsum(dim=-1)
    threshold = float(quantile) * sorted_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    quantile_index = (cumulative >= threshold).to(torch.int64).argmax(dim=-1, keepdim=True)
    return sorted_values.gather(-1, quantile_index).squeeze(-1)


def masked_robust_stats(features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-sample robust statistics for [B,C,H,W] features."""
    if features.dim() == 3:
        features = features.unsqueeze(0)
    if features.dim() != 4:
        raise ValueError(f"Expected features [B,C,H,W], got shape={tuple(features.shape)}")
    mask = _as_mask4d(mask, features.shape[-2:], features.size(0))
    valid = mask.sum(dim=(-2, -1), keepdim=True) >= 1.0
    mask = torch.where(valid, mask, torch.ones_like(mask))

    median = _weighted_quantile(features, mask, 0.5)
    absolute_deviation = (features - median[:, :, None, None]).abs()
    mad = _weighted_quantile(absolute_deviation, mask, 0.5)
    robust_scale = (1.4826 * mad).clamp_min(1e-3)
    normalized_deviation = absolute_deviation / (2.5 * robust_scale[:, :, None, None])
    robust_weight = mask / (1.0 + normalized_deviation.square())
    if robust_weight.size(1) == 1 and features.size(1) != 1:
        robust_weight = robust_weight.expand(-1, features.size(1), -1, -1)
    denominator = robust_weight.sum(dim=(-2, -1)).clamp_min(1e-6)
    mean = (features * robust_weight).sum(dim=(-2, -1)) / denominator
    variance = (
        (features - mean[:, :, None, None]).square() * robust_weight
    ).sum(dim=(-2, -1)) / denominator
    return {
        "mean": mean,
        "std": torch.sqrt(variance.clamp_min(1e-8)),
        "median": median,
        "mad": mad,
    }


def _masked_gaussian_mean(value: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    max_radius = max(min(value.shape[-2:]) - 1, 0)
    radius = min(max(int(radius), 0), max_radius)
    if radius == 0:
        return value
    numerator = gaussian_blur2d(value * mask, radius=radius)
    denominator = gaussian_blur2d(mask, radius=radius).clamp_min(1e-4)
    return numerator / denominator


def _erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=2 * width + 1, stride=1, padding=width)


def build_reference_color_safe_mask(
    reference_image: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    reference_rgb, _ = _to_rgb01(reference_image)
    hair_mask = _as_mask4d(
        reference_hair_mask,
        reference_rgb.shape[-2:],
        reference_rgb.size(0),
    )
    eroded = _erode_mask(hair_mask, config.reference_mask_erode)
    eroded_valid = eroded.sum(dim=(-2, -1), keepdim=True) >= 16.0
    stats_mask = torch.where(eroded_valid, eroded, hair_mask)

    reference_lab = rgb_to_lab(reference_rgb)
    lightness = reference_lab[:, 0:1]
    chroma = torch.linalg.vector_norm(reference_lab[:, 1:3], dim=1, keepdim=True)
    l_stats = masked_robust_stats(lightness, stats_mask)
    c_stats = masked_robust_stats(chroma, stats_mask)
    median_l = l_stats["median"][:, :, None, None]
    robust_mad_l = (1.4826 * l_stats["mad"])[:, :, None, None]

    local_l = _masked_gaussian_mean(lightness, hair_mask, config.highlight_local_radius)
    local_c = _masked_gaussian_mean(chroma, hair_mask, config.highlight_local_radius)
    global_margin = torch.maximum(
        config.highlight_mad_scale * robust_mad_l,
        torch.full_like(robust_mad_l, config.highlight_global_min_margin),
    )
    global_bright = torch.sigmoid(
        (lightness - median_l - global_margin) / max(config.highlight_softness_l, 1e-4)
    )
    local_bright = torch.sigmoid(
        (lightness - local_l - config.highlight_local_l_margin)
        / max(config.highlight_softness_l, 1e-4)
    )
    local_desaturation = torch.sigmoid(
        (local_c - chroma - config.highlight_local_c_margin)
        / max(config.highlight_softness_c, 1e-4)
    )
    chroma_ratio = chroma / local_c.clamp_min(1.0)
    ratio_desaturation = torch.sigmoid(
        (config.highlight_chroma_ratio - chroma_ratio) / 0.08
    )
    desaturation = torch.maximum(local_desaturation, ratio_desaturation)

    near_clip = torch.sigmoid((reference_rgb.amax(dim=1, keepdim=True) - config.highlight_near_clip) / 0.015)
    low_clip_chroma = torch.sigmoid(
        (config.highlight_clip_chroma - chroma) / max(config.highlight_softness_c, 1e-4)
    )
    brightness_confidence = torch.maximum(global_bright, local_bright)
    specular_confidence = brightness_confidence * desaturation
    clip_confidence = near_clip * low_clip_chroma * local_bright
    highlight_confidence = torch.maximum(
        specular_confidence,
        clip_confidence,
    ) * hair_mask

    raw_safe_mask = hair_mask * (1.0 - highlight_confidence.clamp(0, 1))
    hair_area = hair_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    raw_safe_fraction = raw_safe_mask.sum(dim=(-2, -1), keepdim=True) / hair_area
    fallback_mix = (
        (config.min_safe_fraction - raw_safe_fraction)
        / (1.0 - raw_safe_fraction).clamp_min(1e-6)
    ).clamp(0, 1)
    safe_mask = raw_safe_mask * (1.0 - fallback_mix) + hair_mask * fallback_mix
    safe_fraction = safe_mask.sum(dim=(-2, -1), keepdim=True) / hair_area
    rejected_mask = (hair_mask - safe_mask).clamp(0, 1)

    return {
        "safe_ref_mask": safe_mask.clamp(0, 1),
        "rejected_highlight_mask": rejected_mask,
        "highlight_confidence": highlight_confidence.clamp(0, 1),
        "safe_fraction": safe_fraction.flatten(1).mean(dim=1),
        "raw_safe_fraction": raw_safe_fraction.flatten(1).mean(dim=1),
        "reference_lab": reference_lab,
        "reference_rgb01": reference_rgb,
        "median_chroma": c_stats["median"].squeeze(1),
    }


def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if upper <= lower:
        raise ValueError(f"Smoothstep upper={upper} must be greater than lower={lower}")
    x = ((value - lower) / (upper - lower)).clamp(0, 1)
    return x.square() * (3.0 - 2.0 * x)


def compute_color_need_gates(
    ref_mean_ab: torch.Tensor,
    base_mean_ab: torch.Tensor,
    delta_l_global: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    delta_ab = ref_mean_ab - base_mean_ab
    distance_ab = torch.linalg.vector_norm(delta_ab, dim=1)
    chroma_need_gate = _smoothstep(
        distance_ab,
        config.chroma_no_edit_threshold,
        config.chroma_full_edit_threshold,
    )
    lightness_need_gate = _smoothstep(
        delta_l_global.abs(),
        config.lightness_no_edit_threshold,
        config.lightness_full_edit_threshold,
    )
    edit_need_gate = 1.0 - (1.0 - chroma_need_gate) * (1.0 - lightness_need_gate)
    return {
        "chroma_need_gate": chroma_need_gate,
        "lightness_need_gate": lightness_need_gate,
        "edit_need_gate": edit_need_gate,
        "delta_ab": delta_ab,
        "distance_ab": distance_ab,
    }


def compute_color_descriptor(
    reference_lab: torch.Tensor,
    safe_ref_mask: torch.Tensor,
    base_lab: torch.Tensor,
    target_hair_mask: torch.Tensor,
    safe_fraction: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = config or ColorConditionConfigV8()
    ref_ab_stats = masked_robust_stats(reference_lab[:, 1:3], safe_ref_mask)
    ref_l_stats = masked_robust_stats(reference_lab[:, 0:1], safe_ref_mask)
    ref_c_stats = masked_robust_stats(
        torch.linalg.vector_norm(reference_lab[:, 1:3], dim=1, keepdim=True),
        safe_ref_mask,
    )
    base_ab_stats = masked_robust_stats(base_lab[:, 1:3], target_hair_mask)
    base_l_stats = masked_robust_stats(base_lab[:, 0:1], target_hair_mask)
    base_c_stats = masked_robust_stats(
        torch.linalg.vector_norm(base_lab[:, 1:3], dim=1, keepdim=True),
        target_hair_mask,
    )

    ref_ab = ref_ab_stats["mean"]
    base_ab = base_ab_stats["mean"]
    ref_c = ref_c_stats["mean"].squeeze(1)
    base_c = base_c_stats["mean"].squeeze(1)
    ref_l = ref_l_stats["median"].squeeze(1)
    base_l = base_l_stats["median"].squeeze(1)
    delta_l_unclamped = ref_l - base_l
    delta_l = delta_l_unclamped.clamp(-config.max_global_l_shift, config.max_global_l_shift)
    gate_info = compute_color_need_gates(ref_ab, base_ab, delta_l, config)

    ref_hue = ref_ab / torch.linalg.vector_norm(ref_ab, dim=1, keepdim=True).clamp_min(1e-4)
    base_hue = base_ab / torch.linalg.vector_norm(base_ab, dim=1, keepdim=True).clamp_min(1e-4)
    delta_ab = gate_info["delta_ab"]
    descriptor = torch.cat(
        [
            ref_ab / 110.0,
            (ref_c / 110.0).unsqueeze(1),
            (ref_l / 100.0).unsqueeze(1),
            ref_hue,
            ref_ab_stats["std"] / 110.0,
            base_ab / 110.0,
            (base_c / 110.0).unsqueeze(1),
            (base_l / 100.0).unsqueeze(1),
            base_hue,
            delta_ab / 110.0,
            ((ref_c - base_c) / 110.0).unsqueeze(1),
            (delta_l / 100.0).unsqueeze(1),
            (gate_info["distance_ab"] / 110.0).unsqueeze(1),
            safe_fraction.unsqueeze(1),
        ],
        dim=1,
    )
    if descriptor.size(1) != COLOR_DESCRIPTOR_DIM:
        raise RuntimeError(f"Expected descriptor dim={COLOR_DESCRIPTOR_DIM}, got {descriptor.size(1)}")

    metrics = {
        "ref_mean_ab": ref_ab,
        "base_mean_ab": base_ab,
        "ref_median_l": ref_l,
        "base_median_l": base_l,
        "delta_l_global": delta_l,
        "delta_l_unclamped": delta_l_unclamped,
        "ref_base_ab_distance": gate_info["distance_ab"],
        "ref_base_global_l_distance": delta_l.abs(),
        "chroma_need_gate": gate_info["chroma_need_gate"],
        "lightness_need_gate": gate_info["lightness_need_gate"],
        "edit_need_gate": gate_info["edit_need_gate"],
    }
    return descriptor, metrics


def conditional_ab_by_luma(
    target_luma_norm: torch.Tensor,
    ref_luma_norm: torch.Tensor,
    ref_ab: torch.Tensor,
    ref_mask: torch.Tensor,
    bins: int = 9,
) -> torch.Tensor:
    bins = max(int(bins), 3)
    centers = torch.linspace(0.0, 1.0, bins, device=ref_ab.device, dtype=ref_ab.dtype).view(1, bins, 1, 1)
    sigma = 0.55 / max(bins - 1, 1)
    ref_mask = _as_mask4d(ref_mask, ref_ab.shape[-2:], ref_ab.size(0))
    ref_weights = torch.exp(-0.5 * ((ref_luma_norm - centers) / sigma).square()) * ref_mask
    ref_weights_5d = ref_weights.unsqueeze(2)
    denominator = ref_weights_5d.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    bin_ab = (ref_ab.unsqueeze(1) * ref_weights_5d).sum(dim=(-2, -1), keepdim=True) / denominator
    global_ab = masked_robust_stats(ref_ab, ref_mask)["mean"][:, None, :, None, None]
    bin_ab = torch.where(denominator > 8.0, bin_ab, global_ab)
    target_weights = torch.exp(-0.5 * ((target_luma_norm - centers) / sigma).square()).unsqueeze(2)
    return (bin_ab * target_weights).sum(dim=1) / target_weights.sum(dim=1).clamp_min(1e-6)


def build_pseudo_color_target(
    reference_lab: torch.Tensor,
    safe_ref_mask: torch.Tensor,
    base_lab: torch.Tensor,
    target_hair_mask: torch.Tensor,
    delta_l_global: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    target_hair_mask = _as_mask4d(target_hair_mask, base_lab.shape[-2:], base_lab.size(0))
    target_ref_ab = conditional_ab_by_luma(
        target_luma_norm=(base_lab[:, 0:1] / 100.0).clamp(0, 1),
        ref_luma_norm=(reference_lab[:, 0:1] / 100.0).clamp(0, 1),
        ref_ab=reference_lab[:, 1:3],
        ref_mask=safe_ref_mask,
        bins=config.conditional_luma_bins,
    )
    delta_l = delta_l_global[:, None, None, None]
    pseudo_ab = base_lab[:, 1:3] + target_hair_mask * (target_ref_ab - base_lab[:, 1:3])
    pseudo_l = base_lab[:, 0:1] + target_hair_mask * delta_l
    pseudo_lab = torch.cat([pseudo_l.clamp(0, 100), pseudo_ab], dim=1)
    return {
        "pseudo_lab": pseudo_lab,
        "pseudo_rgb01": lab_to_rgb(pseudo_lab),
        "target_ref_ab": target_ref_ab,
    }


def build_reference_color_proxy(
    reference_image: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    reference_lab: torch.Tensor,
    safe_ref_mask: torch.Tensor,
) -> torch.Tensor:
    _, normalized = _to_rgb01(reference_image)
    hair_mask = _as_mask4d(
        reference_hair_mask,
        reference_lab.shape[-2:],
        reference_lab.size(0),
    )
    l_stats = masked_robust_stats(reference_lab[:, 0:1], safe_ref_mask)
    ab_stats = masked_robust_stats(reference_lab[:, 1:3], safe_ref_mask)
    proxy_lab = torch.cat(
        [
            l_stats["median"][:, :, None, None].expand(-1, -1, *reference_lab.shape[-2:]),
            ab_stats["mean"][:, :, None, None].expand(-1, -1, *reference_lab.shape[-2:]),
        ],
        dim=1,
    )
    flat_hair_rgb = lab_to_rgb(proxy_lab)
    neutral_background = torch.full_like(flat_hair_rgb, 0.5)
    proxy_rgb01 = flat_hair_rgb * hair_mask + neutral_background * (1.0 - hair_mask)
    return _from_rgb01(proxy_rgb01, normalized)


def build_color_condition_bundle(
    reference_image: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    base_image: torch.Tensor,
    target_hair_mask: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, object]:
    """Build the shared training/inference color condition for BlendingV8."""
    config = config or ColorConditionConfigV8()
    reference_image = _as_image4d(reference_image)
    base_image = _as_image4d(base_image)
    if reference_image.size(0) != base_image.size(0):
        raise ValueError("Reference and base batch sizes must match")
    if reference_image.shape[-2:] != base_image.shape[-2:]:
        reference_image = F.interpolate(reference_image, size=base_image.shape[-2:], mode="bilinear", align_corners=False)

    safe_info = build_reference_color_safe_mask(reference_image, reference_hair_mask, config)
    base_rgb01, base_normalized = _to_rgb01(base_image)
    base_lab = rgb_to_lab(base_rgb01)
    target_hair_mask = _as_mask4d(target_hair_mask, base_lab.shape[-2:], base_lab.size(0))
    descriptor, metrics = compute_color_descriptor(
        reference_lab=safe_info["reference_lab"],
        safe_ref_mask=safe_info["safe_ref_mask"],
        base_lab=base_lab,
        target_hair_mask=target_hair_mask,
        safe_fraction=safe_info["safe_fraction"],
        config=config,
    )
    pseudo = build_pseudo_color_target(
        reference_lab=safe_info["reference_lab"],
        safe_ref_mask=safe_info["safe_ref_mask"],
        base_lab=base_lab,
        target_hair_mask=target_hair_mask,
        delta_l_global=metrics["delta_l_global"],
        config=config,
    )
    color_proxy = build_reference_color_proxy(
        reference_image,
        reference_hair_mask,
        safe_info["reference_lab"],
        safe_info["safe_ref_mask"],
    )
    metrics = {
        **metrics,
        "safe_fraction": safe_info["safe_fraction"],
        "raw_safe_fraction": safe_info["raw_safe_fraction"],
        "rejected_fraction": 1.0 - safe_info["safe_fraction"],
    }
    return {
        "safe_ref_mask": safe_info["safe_ref_mask"],
        "rejected_highlight_mask": safe_info["rejected_highlight_mask"],
        "descriptor": descriptor,
        "chroma_need_gate": metrics["chroma_need_gate"],
        "lightness_need_gate": metrics["lightness_need_gate"],
        "edit_need_gate": metrics["edit_need_gate"],
        "pseudo_lab": pseudo["pseudo_lab"],
        "pseudo_rgb": _from_rgb01(pseudo["pseudo_rgb01"], base_normalized),
        "color_proxy": color_proxy,
        "target_ref_ab": pseudo["target_ref_ab"],
        "base_lab": base_lab,
        "metrics": metrics,
    }
