"""V2.36 hair-only chroma and illumination appearance transfer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.hair_only_chroma_disentanglement_v835 import build_hair_ownership_v835
from models.hybrid_hair_carrier_v829 import normalized_blur_v829


def _match(value: torch.Tensor, size: tuple[int, int], *, mask: bool = False) -> torch.Tensor:
    if value.dim() == 2:
        value = value[None, None]
    elif value.dim() == 3:
        value = value[:, None]
    if value.shape[-2:] != size:
        value = F.interpolate(
            value.float(),
            size=size,
            mode="nearest" if mask else "bilinear",
            align_corners=None if mask else False,
        )
    return value


def _masked_global_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (value * mask).flatten(2).sum(2, keepdim=True)
    denominator = mask.flatten(2).sum(2, keepdim=True).clamp_min(1e-6)
    return (numerator / denominator).unsqueeze(-1)


class HairAppearanceDecompositionTransferV836(nn.Module):
    """Fuse reference chroma and Target/SATD illumination into a Strong Anchor."""

    def __init__(
        self,
        *,
        chroma_radius: int = 9,
        chroma_residual_radius: int = 3,
        illumination_radius: int = 11,
        chroma_scale: float = 18.0,
        chroma_gain: float = 1.0,
        illumination_gain: float = 0.35,
        max_illumination_shift: float = 20.0,
    ):
        super().__init__()
        if min(chroma_radius, chroma_residual_radius, illumination_radius) < 1:
            raise ValueError("V2.36 blur radii must be positive")
        if chroma_scale <= 0:
            raise ValueError("V2.36 chroma_scale must be positive")
        if not 0.0 <= illumination_gain < chroma_gain:
            raise ValueError("V2.36 gains must satisfy 0 <= Gl < Gc")
        if max_illumination_shift <= 0:
            raise ValueError("V2.36 max_illumination_shift must be positive")
        self.chroma_radius = int(chroma_radius)
        self.chroma_residual_radius = int(chroma_residual_radius)
        self.illumination_radius = int(illumination_radius)
        self.chroma_scale = float(chroma_scale)
        self.chroma_gain = float(chroma_gain)
        self.illumination_gain = float(illumination_gain)
        self.max_illumination_shift = float(max_illumination_shift)

    def config_dict(self) -> dict[str, object]:
        return {
            "chroma_radius": self.chroma_radius,
            "chroma_residual_radius": self.chroma_residual_radius,
            "illumination_radius": self.illumination_radius,
            "chroma_scale": self.chroma_scale,
            "chroma_gain_gc": self.chroma_gain,
            "illumination_gain_gl": self.illumination_gain,
            "max_illumination_shift": self.max_illumination_shift,
            "chroma_owner": "COLOR_REFERENCE_HAIR_LAB_AB",
            "illumination_owner": "TARGET_SATD_HAIR_LOW_FREQUENCY_L",
            "structure_owner": "STRONG_ANCHOR",
            "confidence": "OWNERSHIP_TIMES_0.7_SUPPORT_PLUS_0.3_SIMILARITY",
        }

    @staticmethod
    def _masked_chroma_field(
        reference_rgb: torch.Tensor,
        reference_hair_mask: torch.Tensor,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_ab = rgb_to_lab(reference_rgb)[:, 1:]
        support_mask = reference_hair_mask.clamp(0, 1)
        local_ab, local_support = normalized_blur_v829(reference_ab, support_mask, radius)
        global_ab = _masked_global_mean(reference_ab, support_mask)
        field = torch.where(local_support > 0.05, local_ab, global_ab)
        neutral = torch.full_like(reference_rgb, 0.5)
        hair_only_rgb = reference_rgb * support_mask + neutral * (1.0 - support_mask)
        return field, local_support.clamp(0, 1), hair_only_rgb

    def forward(
        self,
        *,
        strong_anchor_rgb: torch.Tensor,
        color_reference_rgb: torch.Tensor,
        hair_mask: torch.Tensor,
        face_mask: torch.Tensor,
        illumination_rgb: torch.Tensor | None = None,
        target_appearance_rgb: torch.Tensor | None = None,
        satd_rgb: torch.Tensor | None = None,
        anchor_hair_mask: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
        reference_hair_mask: torch.Tensor | None = None,
        illumination_hair_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if illumination_rgb is None:
            illumination_rgb = target_appearance_rgb if target_appearance_rgb is not None else satd_rgb
        if illumination_rgb is None:
            raise ValueError("V2.36 requires Target/SATD illumination_rgb")
        anchor = _match(strong_anchor_rgb, strong_anchor_rgb.shape[-2:]).float().clamp(0, 1)
        size = anchor.shape[-2:]
        reference = _match(color_reference_rgb, size).to(anchor).clamp(0, 1)
        illumination = _match(illumination_rgb, size).to(anchor).clamp(0, 1)
        target_hair = _match(
            hair_mask if target_hair_mask is None else target_hair_mask, size, mask=True
        ).to(anchor).clamp(0, 1)
        anchor_hair = target_hair if anchor_hair_mask is None else _match(
            anchor_hair_mask, size, mask=True
        ).to(anchor).clamp(0, 1)
        reference_hair = target_hair if reference_hair_mask is None else _match(
            reference_hair_mask, size, mask=True
        ).to(anchor).clamp(0, 1)
        illumination_hair = target_hair if illumination_hair_mask is None else _match(
            illumination_hair_mask, size, mask=True
        ).to(anchor).clamp(0, 1)
        face = _match(face_mask, size, mask=True).to(anchor).clamp(0, 1)
        ownership = build_hair_ownership_v835(
            anchor_hair_mask=anchor_hair,
            target_hair_mask=target_hair,
            face_mask=face,
            size=size,
        ).to(anchor)

        anchor_lab = rgb_to_lab(anchor)
        reference_ab, reference_support, hair_only_rgb = self._masked_chroma_field(
            reference, reference_hair, self.chroma_radius
        )
        global_ab = _masked_global_mean(reference_ab, reference_support)
        chroma_distance = torch.linalg.vector_norm(reference_ab - global_ab, dim=1, keepdim=True)
        chroma_similarity = torch.exp(-chroma_distance / self.chroma_scale).clamp(0.20, 1.0)
        support_confidence = reference_support.clamp(0, 1)
        chroma_confidence = (
            ownership * (0.7 * support_confidence + 0.3 * chroma_similarity)
        ).clamp(0, 1)

        raw_delta_ab = reference_ab - anchor_lab[:, 1:]
        smooth_delta_ab, residual_support = normalized_blur_v829(
            raw_delta_ab, ownership, self.chroma_residual_radius
        )
        delta_ab_field = torch.where(residual_support > 0.05, smooth_delta_ab, raw_delta_ab)
        delta_ab = delta_ab_field * ownership

        illumination_l = rgb_to_lab(illumination)[:, :1]
        illumination_l_low, illumination_support = normalized_blur_v829(
            illumination_l, illumination_hair, self.illumination_radius
        )
        anchor_l_low, anchor_l_support = normalized_blur_v829(
            anchor_lab[:, :1], ownership, self.illumination_radius
        )
        valid_illumination = (illumination_support > 0.05) & (anchor_l_support > 0.05)
        raw_illumination_residual = torch.where(
            valid_illumination,
            illumination_l_low - anchor_l_low,
            torch.zeros_like(illumination_l_low),
        ).clamp(-self.max_illumination_shift, self.max_illumination_shift)
        illumination_residual = raw_illumination_residual * ownership
        illumination_confidence = (ownership * illumination_support.clamp(0, 1)).clamp(0, 1)

        chroma_residual = self.chroma_gain * chroma_confidence * delta_ab_field
        gated_illumination_residual = (
            self.illumination_gain * illumination_confidence * raw_illumination_residual
        )
        final_lab = anchor_lab.clone()
        final_lab[:, 1:] = anchor_lab[:, 1:] + chroma_residual
        final_lab[:, :1] = anchor_lab[:, :1] + gated_illumination_residual
        fused_rgb = lab_to_rgb(final_lab).clamp(0, 1)
        # The branch residuals already contain soft ownership. A boolean compose
        # prevents a second confidence multiplication while keeping exact pixels
        # everywhere ownership is zero.
        final_rgb = torch.where((ownership > 0).expand_as(anchor), fused_rgb, anchor)
        if not torch.isfinite(final_rgb).all():
            raise ValueError("V2.36 appearance decomposition produced NaN or Inf")
        if not return_aux:
            return final_rgb

        chroma_map_lab = anchor_lab.clone()
        chroma_map_lab[:, 1:] = anchor_lab[:, 1:] + chroma_residual
        chroma_only_rgb = torch.where(
            (ownership > 0).expand_as(anchor), lab_to_rgb(chroma_map_lab).clamp(0, 1), anchor
        )
        hair_chroma_map = chroma_only_rgb * ownership
        hair_illumination_map = (illumination_l_low / 100.0).clamp(0, 1) * ownership
        final_leakage_map = (final_rgb - anchor).abs().mean(1, keepdim=True) * (1.0 - ownership)
        return final_rgb, {
            "hair_ownership": ownership,
            "final_hair_mask": ownership,
            "color_hair_mask": reference_hair,
            "color_hair_only_rgb": hair_only_rgb,
            "hair_chroma_map": hair_chroma_map,
            "chroma_only_rgb": chroma_only_rgb,
            "hair_illumination_map": hair_illumination_map,
            "illumination_residual": illumination_residual,
            "gated_illumination_residual": gated_illumination_residual,
            "chroma_confidence": chroma_confidence,
            "confidence_map": chroma_confidence,
            "support_confidence": support_confidence,
            "chroma_similarity": chroma_similarity,
            "raw_delta_chroma_ab": raw_delta_ab,
            "delta_chroma_ab": delta_ab,
            "chroma_residual": chroma_residual,
            "illumination_confidence": illumination_confidence,
            "final_leakage_map": final_leakage_map,
            "leakage_map": final_leakage_map,
            "final_rgb": final_rgb,
            "face_rgb_change_max": ((final_rgb - anchor) * face).abs().amax(dim=(1, 2, 3)),
            "non_hair_change_max": (
                (final_rgb - anchor) * (1.0 - ownership)
            ).abs().amax(dim=(1, 2, 3)),
        }


HairAppearanceTransferV836 = HairAppearanceDecompositionTransferV836
HairAppearanceDecompositionV836 = HairAppearanceDecompositionTransferV836

__all__ = [
    "HairAppearanceDecompositionTransferV836",
    "HairAppearanceTransferV836",
    "HairAppearanceDecompositionV836",
]
