from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844, _blur


def test_noop_residual():
    carrier = torch.tensor([[[[50.0]], [[8.0]], [[4.0]]]])
    _, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=_blur(carrier[:, :1]), reference_ab_low=_blur(carrier[:, 1:]), return_aux=True)
    assert float(aux["delta_l_low"].abs().max()) < 1e-5
    assert float(aux["total_gamut_scale"].min()) > 0.95


if __name__ == "__main__":
    test_noop_residual()
    print("V2.44 no-op test: PASS")
