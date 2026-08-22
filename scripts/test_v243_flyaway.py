from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.strand_aware_hair_matte_v843 import StrandAwareHairMatteV843


def test_flyaway_outside_coarse_mask():
    coarse = torch.zeros(1, 1, 32, 32); coarse[:, :, 12:20, 12:20] = 1
    evidence = torch.zeros_like(coarse); evidence[:, :, 10:22, 10:22] = .8
    strand = torch.zeros_like(coarse); strand[:, :, 10:22, 10:22] = .9
    alpha, aux = StrandAwareHairMatteV843()(coarse_target_hair_mask=coarse,
        safe_dense_core=torch.zeros_like(coarse), uncertain_core=torch.zeros_like(coarse),
        anchor_hair_evidence=evidence, strand_structure_confidence=strand,
        source_skin_mask=torch.zeros_like(coarse), return_aux=True)
    assert float(alpha[0, 0, 10:12, 10:22].max()) > 0.1
    assert float(aux["flyaway_alpha"].max()) <= .55


if __name__ == "__main__":
    test_flyaway_outside_coarse_mask(); print("V2.43 flyaway test: PASS")
