"""V2.43 illumination-coupled hair appearance primitives."""

from __future__ import annotations

import torch

from models.reference_chroma_budget_v843 import apply_reference_chroma_budget_v843


def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    x = ((value - float(low)) / max(float(high - low), 1e-6)).clamp(0, 1)
    return x * x * (3.0 - 2.0 * x)


def photometric_chroma_scales_v843(final_l: torch.Tensor) -> dict[str, torch.Tensor]:
    shadow_factor = _smoothstep(final_l, 12.0, 35.0)
    highlight_factor = _smoothstep(final_l, 70.0, 92.0)
    return {
        "shadow_factor": shadow_factor,
        "highlight_factor": highlight_factor,
        "shadow_chroma_scale": 0.30 + 0.70 * shadow_factor,
        "highlight_chroma_scale": 1.0 - 0.45 * highlight_factor,
    }


class HairPhotometricAppearanceV843:
    def __init__(self, *, scene_mix: float = 0.15, realism_enabled: bool = True) -> None:
        self.scene_mix = min(max(float(scene_mix), 0.0), 0.15)
        self.realism_enabled = bool(realism_enabled)

    def __call__(self, *, reference_ab: torch.Tensor, final_l: torch.Tensor,
                 scene_illumination_ab: torch.Tensor | None = None,
                 carrier_chroma: torch.Tensor | None = None,
                 source_hair_l: torch.Tensor | None = None,
                 reference_l: torch.Tensor | None = None,
                 return_aux: bool = False, **_: torch.Tensor):
        scales = photometric_chroma_scales_v843(final_l)
        chroma = torch.linalg.vector_norm(reference_ab.float(), dim=1, keepdim=True)
        unit = reference_ab.float() / chroma.clamp_min(1e-4)
        chroma, chroma_budget = apply_reference_chroma_budget_v843(
            chroma, lightness=final_l, reference_chroma=chroma,
            shadow_scale=scales["shadow_chroma_scale"],
            highlight_scale=scales["highlight_chroma_scale"],
        )
        if carrier_chroma is not None:
            chroma = (chroma + carrier_chroma.float()).clamp_min(0.0)
        if source_hair_l is not None and reference_l is not None and self.realism_enabled:
            bleach_demand = (reference_l.float() - source_hair_l.float()).clamp_min(0.0)
            plausibility = 1.0 - 0.20 * (bleach_demand / 40.0).clamp(0, 1)
            chroma = chroma * plausibility
        else:
            bleach_demand = torch.zeros_like(final_l)
            plausibility = torch.ones_like(final_l)
        scene = torch.zeros_like(reference_ab) if scene_illumination_ab is None else scene_illumination_ab.float()
        if scene.shape[-2:] != final_l.shape[-2:]:
            scene = scene.expand(-1, -1, final_l.shape[-2], final_l.shape[-1])
        mix = self.scene_mix
        final_ab = unit * chroma + mix * scene
        aux = {
            **scales,
            "photometric_chroma": chroma,
            "reference_chroma_budget": chroma_budget,
            "scene_illumination_ab": scene,
            "bleach_demand": bleach_demand,
            "plausibility_scale": plausibility,
            "v243_photometric_realism_enabled": final_l.new_tensor(float(self.realism_enabled)),
            "final_ab": final_ab,
        }
        return (final_ab, aux) if return_aux else final_ab


__all__ = ["HairPhotometricAppearanceV843", "photometric_chroma_scales_v843"]
