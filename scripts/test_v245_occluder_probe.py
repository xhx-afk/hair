from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.occluder_probe_v245 import OccluderProbeV245


def test_occluder_is_frontmost():
    base = torch.zeros(1, 3, 16, 16)
    hair = torch.ones_like(base)
    alpha = torch.ones(1, 1, 16, 16)
    occ = torch.zeros_like(alpha); occ[:, :, 7:9] = 1
    current, protected = OccluderProbeV245.apply(base_rgb=base, hair_rgb=hair, hair_alpha=alpha, occluder_alpha=occ)
    assert float(current[:, :, 7:9].mean()) == 1.0
    assert float(protected[:, :, 7:9].mean()) == 0.0


if __name__ == "__main__":
    test_occluder_is_frontmost(); print("V2.45 occluder probe test: PASS")
