from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.matte_probe_v245 import MatteProbeV245


def test_continuous_alpha_and_flyaway_cap():
    coarse = torch.zeros(1, 1, 32, 32)
    coarse[:, :, 10:22, 10:22] = 1.0
    alpha, aux = MatteProbeV245()(coarse_target_hair_mask=coarse, distance_prior=torch.full_like(coarse, .7), anchor_hair_evidence=torch.full_like(coarse, .8), strand_structure_confidence=torch.full_like(coarse, .8), source_skin_mask=torch.zeros_like(coarse), strong_anchor_rgb=torch.rand(1, 3, 32, 32), return_aux=True)
    assert bool(torch.isfinite(alpha).all())
    assert float(alpha.min()) >= 0 and float(alpha.max()) <= 1
    assert float(aux["flyaway_alpha"].max()) <= .450001


if __name__ == "__main__":
    test_continuous_alpha_and_flyaway_cap(); print("V2.45 matte probe test: PASS")
