from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.scene_illumination_estimator_v843 import SceneIlluminationEstimatorV843


def test_scene_illumination_is_restrained():
    hair = torch.zeros(1, 1, 32, 32); hair[:, :, 10:22, 10:22] = 1
    rgb = torch.full((1, 3, 32, 32), .5)
    out = SceneIlluminationEstimatorV843()(target_rgb=rgb, target_hair_mask=hair, return_aux=True)
    assert float(out["scene_mix"].max()) <= .15
    assert out["scene_illumination_ab"].shape == (1, 2, 1, 1)


if __name__ == "__main__":
    test_scene_illumination_is_restrained(); print("V2.43 scene illumination test: PASS")
