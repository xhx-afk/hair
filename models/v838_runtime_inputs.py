"""V2.38 runtime contract separating topology, face, skin, and final matte."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask(value: torch.Tensor, reference: torch.Tensor, *, mode: str = "nearest") -> torch.Tensor:
    if value.dim() == 2:
        value = value[None, None]
    elif value.dim() == 3:
        value = value[:, None]
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode=mode)
    return value.to(reference).float().clamp(0, 1)


def _rgb(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return value.to(reference).float().clamp(0, 1)


def build_v838_runtime_inputs(
    *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
    color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
    coarse_target_hair_mask: torch.Tensor, source_face_mask: torch.Tensor,
    source_skin_mask: torch.Tensor, reference_hair_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    for name, value in {
        "Base": base_rgb, "Strong Anchor": strong_anchor_rgb,
        "color reference": color_reference_rgb, "Target/SATD illumination": target_illumination_rgb,
    }.items():
        if value.dim() != 4 or value.size(1) != 3 or value.size(0) != base_rgb.size(0):
            raise ValueError(f"V2.38 {name} must be matching BCHW RGB")
    ref = base_rgb[:, :1]
    return {
        "base_rgb": base_rgb.float().clamp(0, 1),
        "strong_anchor_rgb": _rgb(strong_anchor_rgb, base_rgb),
        "color_reference_rgb": _rgb(color_reference_rgb, base_rgb),
        "target_illumination_rgb": _rgb(target_illumination_rgb, base_rgb),
        "coarse_target_hair_mask": _mask(coarse_target_hair_mask, ref),
        "source_face_mask": _mask(source_face_mask, ref),
        "source_skin_mask": _mask(source_skin_mask, ref),
        "reference_hair_mask": _mask(reference_hair_mask, ref),
    }


__all__ = ["build_v838_runtime_inputs"]
