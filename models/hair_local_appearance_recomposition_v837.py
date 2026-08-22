"""V2.37 end-to-end local appearance transfer and recomposition."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.hair_local_recomposition_v837 import HairLocalRecompositionV837
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from models.reference_hair_chroma_palette_v837 import ReferenceHairChromaPaletteV837
from models.target_relative_chroma_field_v837 import TargetRelativeChromaFieldV837


class HairLocalAppearanceRecompositionV837:
    """Palette -> target-relative chroma -> target L/anchor HF -> Base composition."""

    def __init__(
        self, *, palette_mad_scale: float = 3.5, palette_min_support: int = 16,
        illumination_radius: int = 11, anchor_hf_gain: float = 0.9,
        face_guard: float = 1.0,
    ) -> None:
        if not 0 <= anchor_hf_gain <= 1.5:
            raise ValueError("V2.37 anchor_hf_gain must be in [0, 1.5]")
        self.palette = ReferenceHairChromaPaletteV837(
            mad_scale=palette_mad_scale, min_support=palette_min_support
        )
        self.relative_field = TargetRelativeChromaFieldV837(low_radius=illumination_radius)
        self.recomposer = HairLocalRecompositionV837(face_guard=face_guard)
        self.anchor_hf_gain = float(anchor_hf_gain)
        self.illumination_radius = int(illumination_radius)

    def config_dict(self) -> dict[str, object]:
        return {
            "version": "v2.37",
            "non_hair_owner": "BASE_SOURCE_PRESERVED",
            "hair_structure_owner": "STRONG_ANCHOR_HAIR_ONLY",
            "chroma_owner": "REFERENCE_HAIR_ROBUST_PALETTE",
            "spatial_mapping": "TARGET_RELATIVE_LUMINANCE",
            "illumination_owner": "TARGET_SATD_LOW_FREQUENCY_L",
            "anchor_hf_gain": self.anchor_hf_gain,
            "illumination_radius": self.illumination_radius,
        }

    def __call__(
        self, *, base_rgb: torch.Tensor, strong_anchor_rgb: torch.Tensor,
        color_reference_rgb: torch.Tensor, target_illumination_rgb: torch.Tensor,
        target_hair_mask: torch.Tensor, face_mask: torch.Tensor,
        reference_hair_mask: torch.Tensor, target_hair_alpha: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        # Ownership is computed from target topology and explicitly excludes source face.
        ownership = (target_hair_mask.float().clamp(0, 1) * (1.0 - face_mask.float().clamp(0, 1))).clamp(0, 1)
        palette = self.palette(
            reference_rgb=color_reference_rgb, reference_hair_mask=reference_hair_mask
        )
        field = self.relative_field(
            target_illumination_rgb=target_illumination_rgb,
            target_hair_mask=ownership,
            target_hair_alpha=target_hair_alpha,
            palette=palette,
        )
        anchor_lab = rgb_to_lab(strong_anchor_rgb)
        anchor_l_low = gaussian_blur_v829(anchor_lab[:, :1], self.illumination_radius)
        anchor_hf_l = anchor_lab[:, :1] - anchor_l_low
        target_l = field["target_l_low"] + self.anchor_hf_gain * anchor_hf_l
        hair_lab = torch.cat((target_l, field["target_ab"]), dim=1)
        new_hair_rgb = lab_to_rgb(hair_lab).clamp(0, 1)
        alpha = field["target_hair_alpha"] * ownership * field["chroma_confidence"].clamp(0, 1)
        # Palette reliability should not create local holes: it is global, while alpha
        # remains target-owned. Keep a reliable palette at full core strength.
        alpha = torch.where(ownership > 0, field["target_hair_alpha"] * ownership, alpha)
        final_rgb, recomposition = self.recomposer(
            base_rgb=base_rgb, new_hair_rgb=new_hair_rgb,
            hair_alpha=alpha, face_mask=face_mask,
            target_hair_mask=ownership, return_aux=True,
        )
        if not return_aux:
            return final_rgb
        anchor_ab = anchor_lab[:, 1:]
        target_ab = field["target_ab"]
        chroma_strength = (
            torch.linalg.vector_norm(field["target_ab"] - anchor_ab, dim=1, keepdim=True)
            / torch.linalg.vector_norm(target_ab - anchor_ab, dim=1, keepdim=True).clamp_min(1e-4)
        ).clamp(0, 1)
        undertransfer = recomposition["hair_core"] * (chroma_strength < 0.5).float()
        non_hair_mask = (1.0 - recomposition["hair_alpha"]).clamp(0, 1)
        aux = {
            **palette,
            **field,
            **recomposition,
            "hair_ownership": ownership,
            "reference_hair_mask": reference_hair_mask,
            "target_hair_mask": target_hair_mask,
            "new_hair_rgb": new_hair_rgb,
            "anchor_high_frequency_l": anchor_hf_l,
            "target_l_final": target_l,
            "target_chroma_field": field["target_ab"],
            "chroma_strength": chroma_strength,
            "hair_undertransfer_map": undertransfer,
            "non_hair_mask": non_hair_mask,
            "final_rgb": final_rgb,
            "final_minus_base": final_rgb - base_rgb,
            "non_hair_leakage_map": (final_rgb - base_rgb).abs().mean(1, keepdim=True) * non_hair_mask,
            "face_rgb_change_from_base": ((final_rgb - base_rgb) * face_mask).abs().amax(dim=(1, 2, 3)),
            "non_hair_rgb_change_from_base": ((final_rgb - base_rgb) * non_hair_mask).abs().amax(dim=(1, 2, 3)),
        }
        return final_rgb, aux


__all__ = ["HairLocalAppearanceRecompositionV837"]
