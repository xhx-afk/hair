from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844


def test_l_detail_preserved_under_low_residual():
    yy = torch.linspace(-1, 1, 32).view(1, 1, 32, 1)
    xx = torch.linspace(-1, 1, 32).view(1, 1, 1, 32)
    detail = 3.0 * torch.sin(xx * 18.0) * torch.cos(yy * 11.0)
    carrier = torch.cat((50.0 + 4.0 * xx + detail, torch.zeros_like(detail), torch.zeros_like(detail)), dim=1)
    desired_l_low = torch.full_like(detail, 58.0)
    out, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=desired_l_low, reference_ab_low=torch.zeros(1, 2, 32, 32), return_aux=True)
    assert float(aux["delta_l_low"].abs().max()) <= 12.0001
    assert float((aux["final_l_detail"] - aux["carrier_l_detail"]).abs().mean()) < 0.15
    assert out.shape == (1, 3, 32, 32)


if __name__ == "__main__":
    test_l_detail_preserved_under_low_residual()
    print("V2.44 L detail test: PASS")
