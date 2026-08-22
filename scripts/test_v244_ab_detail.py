from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844


def test_ab_detail_preserved():
    yy = torch.linspace(-1, 1, 32).view(1, 1, 32, 1)
    xx = torch.linspace(-1, 1, 32).view(1, 1, 1, 32)
    detail = 2.0 * torch.sin(xx * 17.0) * torch.cos(yy * 9.0)
    carrier = torch.cat((torch.full_like(detail, 55.0), 12.0 + detail, 5.0 - detail), dim=1)
    _, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=carrier[:, :1], reference_ab_low=carrier[:, 1:], return_aux=True)
    assert float((aux["final_ab_detail"] - aux["carrier_ab_detail"]).abs().mean()) < 0.35


if __name__ == "__main__":
    test_ab_detail_preserved()
    print("V2.44 AB detail test: PASS")
