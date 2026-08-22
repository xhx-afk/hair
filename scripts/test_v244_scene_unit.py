from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_residual_v844 import HairPhotometricResidualV844


def test_scene_residual_is_capped():
    carrier = torch.tensor([[[[55.0]], [[12.0]], [[4.0]]]])
    scene = torch.tensor([[[[100.0]], [[100.0]]]])
    _, aux = HairPhotometricResidualV844()(carrier_lab=carrier, desired_l_low=carrier[:, :1], reference_ab_low=carrier[:, 1:], scene_illumination_ab=scene, return_aux=True)
    assert float(aux["scene_delta_ab"].norm()) <= 5.0001


if __name__ == "__main__":
    test_scene_residual_is_capped()
    print("V2.44 scene residual test: PASS")
