from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.strand_aware_hair_matte_v843 import StrandAwareHairMatteV843


def test_uncertain_transition_remains_bounded():
    coarse = torch.zeros(1, 1, 32, 32)
    coarse[:, :, 8:24, 8:24] = 1.0
    uncertain = torch.zeros_like(coarse)
    uncertain[:, :, 8:24, 8:24] = 1.0
    evidence = torch.full_like(coarse, 0.55)
    strand = torch.full_like(coarse, 0.60)
    alpha, aux = StrandAwareHairMatteV843()(coarse_target_hair_mask=coarse,
        safe_dense_core=torch.zeros_like(coarse), uncertain_core=uncertain,
        anchor_hair_evidence=evidence, strand_structure_confidence=strand,
        source_skin_mask=torch.zeros_like(coarse), return_aux=True)
    assert bool(torch.isfinite(alpha).all())
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0
    assert float(aux["transition_alpha"].max()) <= 1.0


if __name__ == "__main__":
    test_uncertain_transition_remains_bounded()
    print("V2.44 uncertain transition test: PASS")
