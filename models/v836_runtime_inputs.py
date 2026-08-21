"""V2.36 hair appearance decomposition runtime contract."""

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


def _validate_rgb(name: str, value: torch.Tensor, batch: int | None = None) -> None:
    if value.dim() != 4 or value.size(1) != 3:
        raise ValueError(f"V2.36 {name} must be BCHW RGB")
    if batch is not None and value.size(0) != batch:
        raise ValueError(f"V2.36 {name} batch must match Strong Anchor batch")


def build_v836_runtime_inputs(
    *,
    strong_anchor_rgb: torch.Tensor,
    color_reference_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor,
    face_mask: torch.Tensor,
    illumination_rgb: torch.Tensor | None = None,
    target_appearance_rgb: torch.Tensor | None = None,
    satd_rgb: torch.Tensor | None = None,
    anchor_hair_mask: torch.Tensor | None = None,
    reference_hair_mask: torch.Tensor | None = None,
    illumination_hair_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    _validate_rgb("Strong Anchor", strong_anchor_rgb)
    _validate_rgb("color reference", color_reference_rgb, strong_anchor_rgb.size(0))
    if illumination_rgb is None:
        illumination_rgb = target_appearance_rgb if target_appearance_rgb is not None else satd_rgb
    if illumination_rgb is None:
        raise ValueError("V2.36 requires Target/SATD illumination_rgb")
    _validate_rgb("Target/SATD illumination", illumination_rgb, strong_anchor_rgb.size(0))
    reference = strong_anchor_rgb[:, :1]
    runtime = {
        "strong_anchor_rgb": strong_anchor_rgb.float().clamp(0, 1),
        "color_reference_rgb": color_reference_rgb.float().clamp(0, 1),
        "illumination_rgb": illumination_rgb.float().clamp(0, 1),
        "target_hair_mask": _mask(target_hair_mask, reference),
        "face_mask": _mask(face_mask, reference),
    }
    runtime["hair_mask"] = runtime["target_hair_mask"]
    if anchor_hair_mask is not None:
        runtime["anchor_hair_mask"] = _mask(anchor_hair_mask, reference)
    if reference_hair_mask is not None:
        runtime["reference_hair_mask"] = _mask(reference_hair_mask, reference)
    if illumination_hair_mask is not None:
        runtime["illumination_hair_mask"] = _mask(illumination_hair_mask, reference)
    return runtime


__all__ = ["build_v836_runtime_inputs"]
