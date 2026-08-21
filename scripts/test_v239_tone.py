"""Synthetic dark/gray/saturated tone contracts for V2.39."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import rgb_to_lab
from models.hair_local_recomposition_v839 import HairLocalRecompositionV839
from models.v839_runtime_inputs import build_v839_runtime_inputs


def _run(reference_color: tuple[float, float, float], target_color: tuple[float, float, float]):
    h = w = 64
    base = torch.tensor(target_color).view(1, 3, 1, 1).expand(1, 3, h, w).clone()
    reference = torch.tensor(reference_color).view(1, 3, 1, 1).expand_as(base).clone()
    mask = torch.zeros(1, 1, h, w)
    mask[..., 8:56, 8:56] = 1.0
    runtime = build_v839_runtime_inputs(
        base_rgb=base, strong_anchor_rgb=base, color_reference_rgb=reference,
        target_illumination_rgb=base, coarse_target_hair_mask=mask,
        source_face_mask=torch.zeros_like(mask), source_skin_mask=torch.zeros_like(mask),
        reference_hair_mask=mask,
    )
    return HairLocalRecompositionV839()(return_aux=True, **runtime)


def main() -> None:
    dark, dark_aux = _run((0.12, 0.055, 0.025), (0.75, 0.75, 0.75))
    gray, gray_aux = _run((0.20, 0.20, 0.20), (0.78, 0.78, 0.78))
    pink, pink_aux = _run((0.90, 0.18, 0.45), (0.40, 0.40, 0.40))
    dark_l = rgb_to_lab(dark)[:, 0][..., 24:40, 24:40].median()
    gray_l = rgb_to_lab(gray)[:, 0][..., 24:40, 24:40].median()
    pink_lab = rgb_to_lab(pink)
    assert float(dark_l) < 35.0, "dark brown was lifted to target brightness"
    assert float(gray_l) < 45.0, "gray tone lost its intrinsic dark/medium level"
    assert float(pink_lab[:, 1:, 24:40, 24:40].norm(dim=1).median()) > 25.0, "high-chroma color regressed"
    assert float(dark_aux["tone_weight"].mean()) > 0.7
    assert float(gray_aux["tone_weight"].mean()) > 0.7
    assert torch.isfinite(pink).all()
    print("V2.39 intrinsic tone tests: PASS")


if __name__ == "__main__":
    main()
