"""V2.37 palette, geometry-mismatch, and non-hair identity contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_local_appearance_recomposition_v837 import HairLocalAppearanceRecompositionV837
from models.v837_runtime_inputs import build_v837_runtime_inputs


def main() -> None:
    torch.manual_seed(237)
    h = w = 64
    base = torch.zeros(1, 3, h, w)
    base[:, 0] = 0.08
    base[:, 1] = 0.20
    base[:, 2] = 0.75  # Base background must survive exactly.
    anchor = torch.full_like(base, 0.08)
    anchor[:, 0, 8:56, 8:56] = 0.80  # Deliberately contaminated Strong Anchor.
    anchor[:, 1, 8:56, 8:56] = 0.05
    reference = torch.full_like(base, 0.05)
    ref_mask = torch.zeros(1, 1, h, w)
    ref_mask[..., 8:56, :30] = 1.0  # Reference hair on the left half.
    reference[..., 8:56, :30] = torch.tensor([0.92, 0.32, 0.58]).view(1, 3, 1, 1)
    # Sparse green outliers must not change the robust pink palette.
    reference[:, :, 12:14, 10:14] = torch.tensor([0.05, 0.95, 0.05]).view(1, 3, 1, 1)
    target_mask = torch.zeros_like(ref_mask)
    target_mask[..., 8:56, 34:62] = 1.0  # Geometry intentionally mismatched.
    illumination = base.clone()
    illumination[..., 8:56, 34:62] = 0.42
    face = torch.zeros_like(target_mask)

    runtime = build_v837_runtime_inputs(
        base_rgb=base, strong_anchor_rgb=anchor, color_reference_rgb=reference,
        target_illumination_rgb=illumination, target_hair_mask=target_mask,
        face_mask=face, reference_hair_mask=ref_mask,
    )
    model = HairLocalAppearanceRecompositionV837(
        palette_mad_scale=3.5, palette_min_support=16, illumination_radius=5,
        anchor_hf_gain=0.9,
    )
    final, aux = model(return_aux=True, **runtime)
    assert torch.isfinite(final).all()
    assert float((final * (1 - target_mask) - base * (1 - target_mask)).abs().max()) < 1e-7
    assert float((final - base).abs().mean()) > 0.01
    assert float(aux["hair_core"].mean()) > 0.20
    assert float(aux["hair_undertransfer_map"].mean()) < 0.20
    assert float(aux["palette_reliability"].min()) > 0.5
    # The target right half receives one coherent palette despite zero reference support there.
    target_pixels = final[..., 20:44, 40:56]
    assert float(target_pixels[:, 0].mean()) > float(target_pixels[:, 1].mean())
    assert float(target_pixels[:, 2].mean()) > float(target_pixels[:, 1].mean())
    assert float(aux["face_rgb_change_from_base"].max()) < 1e-7
    assert float(aux["non_hair_rgb_change_from_base"].max()) < 1e-7
    print("V2.37 palette and local recomposition tests: PASS")


if __name__ == "__main__":
    main()
