from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844


def test_highlight_scale_is_bounded():
    carrier = torch.tensor([[[[90.0]], [[25.0]], [[5.0]]]])
    _, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=carrier[:, :1], reference_ab_low=carrier[:, 1:], return_aux=True)
    assert 0.54 <= float(aux["highlight_chroma_scale"]) <= 1.0


if __name__ == "__main__":
    test_highlight_scale_is_bounded()
    print("V2.44 highlight residual test: PASS")
