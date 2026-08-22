"""V2.37 hair-local appearance runtime contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.dim() == 2:
        value = value[None, None]
    elif value.dim() == 3:
        value = value[:, None]
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode="nearest")
    return value.to(reference).clamp(0, 1)


def _rgb(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return value.to(reference).float().clamp(0, 1)


def build_v837_runtime_inputs(
    *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
    color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor, face_mask: torch.Tensor,
    reference_hair_mask: torch.Tensor | None = None,
    target_hair_alpha: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    for name, value in {
        "Base": base_rgb, "Strong Anchor": strong_anchor_rgb,
        "color reference": color_reference_rgb, "Target/SATD illumination": target_illumination_rgb,
    }.items():
        if value.dim() != 4 or value.size(1) != 3:
            raise ValueError(f"V2.37 {name} must be BCHW RGB")
        if value.size(0) != base_rgb.size(0):
            raise ValueError(f"V2.37 {name} batch must match Base")
    ref = base_rgb[:, :1]
    runtime = {
        "base_rgb": base_rgb.float().clamp(0, 1),
        "strong_anchor_rgb": _rgb(strong_anchor_rgb, base_rgb),
        "color_reference_rgb": _rgb(color_reference_rgb, base_rgb),
        "target_illumination_rgb": _rgb(target_illumination_rgb, base_rgb),
        "target_hair_mask": _mask(target_hair_mask, ref),
        "face_mask": _mask(face_mask, ref),
    }
    runtime["reference_hair_mask"] = _mask(
        target_hair_mask if reference_hair_mask is None else reference_hair_mask, ref
    )
    runtime["target_hair_alpha"] = _mask(
        target_hair_mask if target_hair_alpha is None else target_hair_alpha, ref
    )
    return runtime


__all__ = ["build_v837_runtime_inputs"]
