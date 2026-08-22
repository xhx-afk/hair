from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_core_confidence_v843 import HairCoreConfidenceV843


def test_coarse_core_skin_false_positive():
    target = torch.zeros(1, 1, 32, 32); target[:, :, 8:24, 8:24] = 1
    rgb = torch.rand(1, 3, 32, 32)
    aux = HairCoreConfidenceV843()(coarse_target_hair_mask=target,
        source_skin_mask=torch.ones_like(target), strong_anchor_rgb=rgb,
        anchor_hair_evidence=torch.full_like(target, 0.1), return_aux=True)
    assert float(aux["safe_dense_core"].sum()) == 0.0


if __name__ == "__main__":
    test_coarse_core_skin_false_positive(); print("V2.43 confidence-core skin test: PASS")
