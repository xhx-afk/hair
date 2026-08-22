"""Synthetic V2.36 hair appearance decomposition contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_appearance_decomposition_v836 import HairAppearanceDecompositionTransferV836
from models.v836_runtime_inputs import build_v836_runtime_inputs


def main() -> None:
    torch.manual_seed(236)
    h = w = 64
    anchor = torch.full((1, 3, h, w), 0.20)
    hair = torch.zeros(1, 1, h, w)
    hair[..., 12:52, 10:54] = 1.0
    # Strong Anchor owns structure and has a deterministic high-frequency stripe.
    stripe = torch.arange(10, 54).view(1, 1, 1, -1).float().remainder(4.0) / 20.0
    anchor[:, 0:1, 12:52, 10:54] = 0.34 + stripe
    anchor[:, 1:2, 12:52, 10:54] = 0.16
    anchor[:, 2:3, 12:52, 10:54] = 0.08

    reference = torch.full_like(anchor, 0.12)
    reference[:, :, 12:52, 10:54] = torch.tensor([0.12, 0.42, 0.70]).view(1, 3, 1, 1)
    # Add local reference variation to exercise residual-only smoothing.
    reference[:, 1:2, 20:44, 20:44] += torch.randn(1, 1, 24, 24) * 0.025
    illumination = torch.full_like(anchor, 0.20)
    illumination[:, :, 12:52, 10:54] = 0.48
    illumination[:, :, 26:32, 20:48] = 0.78  # low-frequency highlight source
    face = torch.zeros_like(hair)
    face[..., 12:18, 10:54] = 1.0

    runtime = build_v836_runtime_inputs(
        strong_anchor_rgb=anchor,
        color_reference_rgb=reference,
        illumination_rgb=illumination,
        target_hair_mask=hair,
        anchor_hair_mask=hair,
        reference_hair_mask=hair,
        illumination_hair_mask=hair,
        face_mask=face,
    )
    model = HairAppearanceDecompositionTransferV836(
        chroma_radius=5,
        chroma_residual_radius=3,
        illumination_radius=7,
        chroma_gain=1.0,
        illumination_gain=0.35,
    )
    final, aux = model(return_aux=True, **runtime)
    assert torch.isfinite(final).all()
    required = {
        "hair_chroma_map", "hair_illumination_map", "illumination_residual",
        "chroma_confidence", "final_leakage_map",
    }
    assert required.issubset(aux)
    expected_confidence = aux["hair_ownership"] * (
        0.7 * aux["support_confidence"] + 0.3 * aux["chroma_similarity"]
    )
    assert torch.allclose(aux["chroma_confidence"], expected_confidence, atol=1e-6)
    assert model.chroma_gain > model.illumination_gain
    # Face and background are identity-locked, including Lab round-trip pixels.
    assert float((final * (1.0 - aux["hair_ownership"]) - anchor * (1.0 - aux["hair_ownership"])).abs().max()) < 1e-7
    assert float((final[..., 12:18, 10:54] - anchor[..., 12:18, 10:54]).abs().max()) < 1e-7
    assert float(aux["illumination_residual"].abs().mean()) > 0.1
    assert float((aux["illumination_residual"] * (1.0 - aux["hair_ownership"])).abs().max()) < 1e-7
    assert float((aux["delta_chroma_ab"] * (1.0 - aux["hair_ownership"])).abs().max()) < 1e-7
    assert float((aux["hair_chroma_map"] * (1.0 - aux["hair_ownership"])).abs().max()) < 1e-7
    # The residual constraint smooths only delta_AB, while preserving ownership.
    raw_hf = aux["raw_delta_chroma_ab"] - F.avg_pool2d(aux["raw_delta_chroma_ab"], 5, 1, 2)
    smooth_hf = aux["delta_chroma_ab"] - F.avg_pool2d(aux["delta_chroma_ab"], 5, 1, 2)
    interior = aux["hair_ownership"][..., 20:44, 20:44]
    assert float((smooth_hf[..., 20:44, 20:44].abs() * interior).mean()) < float(
        (raw_hf[..., 20:44, 20:44].abs() * interior).mean()
    )
    assert float(aux["face_rgb_change_max"].max()) < 1e-7
    assert float(aux["non_hair_change_max"].max()) < 1e-7
    print("V2.36 synthetic hair appearance decomposition tests: PASS")


if __name__ == "__main__":
    main()
