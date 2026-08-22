"""V2.41.2 illumination must not be double-counted."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_local_recomposition_v841 import HairLocalRecompositionV841


def main() -> None:
    h = w = 48
    mask = torch.zeros(1, 1, h, w); mask[..., 8:40, 8:40] = 1
    base = torch.full((1, 3, h, w), 0.35)
    anchor = base.clone()
    reference = torch.full_like(base, 0.22); reference[:, 0] = 0.40
    target = base.clone()
    target[..., 12:28, 12:28] += 0.08
    out, aux = HairLocalRecompositionV841()(base_rgb=base, strong_anchor_rgb=anchor,
        color_reference_rgb=reference, target_illumination_rgb=target,
        coarse_target_hair_mask=mask, source_face_mask=torch.zeros_like(mask),
        source_skin_mask=torch.zeros_like(mask), reference_hair_mask=mask, return_aux=True)
    assert torch.isfinite(out).all()
    assert "deep_uncertain_skin_overlap" in aux
    gain = float(aux["illumination_effective_gain"].mean())
    assert 0.0 < gain < 1.5, gain
    print("V2.41.2 illumination gain tests: PASS")


if __name__ == "__main__":
    main()
