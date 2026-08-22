from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_local_recomposition_v844 import _independent_flyaway_candidate


def test_independent_candidate_is_hair_support_only():
    anchor = torch.zeros(1, 3, 32, 32)
    anchor[:, :, 10:22, 10:22] = 1.0
    coarse = torch.zeros(1, 1, 32, 32)
    coarse[:, :, 12:20, 12:20] = 1.0
    support = torch.zeros_like(coarse)
    support[:, :, 8:24, 8:24] = 1.0
    strand = torch.ones_like(coarse)
    candidate = _independent_flyaway_candidate(anchor, coarse, support, strand)
    assert float(candidate[:, :, :8].max()) == 0.0
    assert float(candidate[:, :, 10:12, 10:22].max()) > 0.0


if __name__ == "__main__":
    test_independent_candidate_is_hair_support_only()
    print("V2.44 hair-only candidate test: PASS")
