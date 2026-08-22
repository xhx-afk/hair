from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.appearance_probe_v245 import AppearanceProbeV245


def test_variants_share_trusted_alpha():
    mask = torch.zeros(1, 1, 32, 32); mask[:, :, 8:24, 8:24] = 1
    evidence = torch.ones_like(mask)
    trusted = AppearanceProbeV245().trusted_core(mask, evidence)
    assert float(trusted.max()) == 1.0
    assert float(trusted[:, :, :8].max()) == 0.0


if __name__ == "__main__":
    test_variants_share_trusted_alpha(); print("V2.45 appearance probe test: PASS")
