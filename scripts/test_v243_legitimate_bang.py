from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_core_confidence_v843 import HairCoreConfidenceV843
from models.strand_aware_hair_matte_v843 import StrandAwareHairMatteV843


def test_legitimate_dense_bang():
    target = torch.ones(1, 1, 32, 32)
    y = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1).expand_as(target)
    rgb = (y / 32.0).repeat(1, 3, 1, 1)
    evidence = torch.full_like(target, 0.9)
    confidence = HairCoreConfidenceV843()(coarse_target_hair_mask=target,
        source_skin_mask=torch.ones_like(target), strong_anchor_rgb=rgb,
        anchor_hair_evidence=evidence, return_aux=True)
    alpha = StrandAwareHairMatteV843()(coarse_target_hair_mask=target,
        safe_dense_core=confidence["safe_dense_core"], uncertain_core=confidence["uncertain_core"],
        transition_prior=confidence["transition_prior"], anchor_hair_evidence=evidence,
        strand_structure_confidence=torch.full_like(target, .8), source_skin_mask=torch.ones_like(target))
    assert float(confidence["safe_dense_core"].mean()) > 0.5
    assert float(alpha.mean()) >= 0.85


if __name__ == "__main__":
    test_legitimate_dense_bang(); print("V2.43 legitimate bang test: PASS")
