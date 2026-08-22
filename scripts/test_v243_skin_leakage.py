from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.face_hair_matte_guard_v843 import FaceHairMatteGuardV843


def test_skin_leakage_is_suppressed():
    shape = (1, 1, 16, 16); alpha = torch.ones(shape)
    guarded = FaceHairMatteGuardV843()(alpha=alpha, source_skin_mask=alpha,
        dense_core_confidence=torch.zeros(shape), strand_structure_confidence=torch.zeros(shape),
        anchor_hair_evidence=torch.zeros(shape))
    assert float(guarded.mean()) < 0.2


if __name__ == "__main__":
    test_skin_leakage_is_suppressed(); print("V2.43 skin leakage test: PASS")
