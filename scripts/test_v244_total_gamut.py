from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844


def test_total_gamut_scale_is_finite_and_bounded():
    carrier = torch.tensor([[[[60.0]], [[140.0]], [[120.0]]]])
    rgb, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=carrier[:, :1], reference_ab_low=carrier[:, 1:], return_aux=True)
    assert bool(torch.isfinite(rgb).all())
    assert float(aux["total_gamut_scale"].min()) >= 0.0
    assert float(aux["total_gamut_scale"].max()) <= 1.0


if __name__ == "__main__":
    test_total_gamut_scale_is_finite_and_bounded()
    print("V2.44 total gamut test: PASS")
