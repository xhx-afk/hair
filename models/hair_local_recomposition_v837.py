"""Hair-local final composition with Base/Source ownership outside hair."""

from __future__ import annotations

import torch


class HairLocalRecompositionV837:
    def __init__(self, *, face_guard: float = 1.0) -> None:
        if not 0 <= face_guard <= 1:
            raise ValueError("V2.37 face_guard must be in [0, 1]")
        self.face_guard = float(face_guard)

    def __call__(
        self, *, base_rgb: torch.Tensor, new_hair_rgb: torch.Tensor,
        hair_alpha: torch.Tensor, face_mask: torch.Tensor,
        target_hair_mask: torch.Tensor, return_aux: bool = False,
    ):
        alpha = (hair_alpha.float().clamp(0, 1) * target_hair_mask.float().clamp(0, 1))
        alpha = alpha * (1.0 - self.face_guard * face_mask.float().clamp(0, 1))
        final = alpha * new_hair_rgb + (1.0 - alpha) * base_rgb
        if not torch.isfinite(final).all():
            raise ValueError("V2.37 hair-local recomposition produced NaN or Inf")
        if not return_aux:
            return final
        return final, {
            "hair_alpha": alpha,
            "hair_core": (alpha >= 0.85).float(),
            "hair_transition": ((alpha > 0) & (alpha < 0.85)).float(),
            "non_hair_mask": (alpha <= 0).float(),
        }


__all__ = ["HairLocalRecompositionV837"]
