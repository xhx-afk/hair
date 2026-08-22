from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844


def test_shadow_scale_is_desaturating():
    carrier = torch.tensor([[[[15.0]], [[30.0]], [[0.0]]]])
    _, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=carrier[:, :1], reference_ab_low=carrier[:, 1:], return_aux=True)
    assert float(aux["shadow_chroma_scale"]) < 0.7


if __name__ == "__main__":
    test_shadow_scale_is_desaturating()
    print("V2.44 shadow residual test: PASS")
