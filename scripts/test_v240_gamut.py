"""Synthetic local gamut compression contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gamut_safe_lab_v840 import gamut_safe_lab_to_rgb_v840, lab_to_rgb_unclamped_v840


def main() -> None:
    lab = torch.zeros(1, 3, 8, 8)
    lab[:, 0] = 55.0
    lab[:, 1] = 100.0
    lab[:, 2] = 80.0
    raw = lab_to_rgb_unclamped_v840(lab)
    safe, scale, clip = gamut_safe_lab_to_rgb_v840(lab)
    assert bool(((raw < 0) | (raw > 1)).any())
    assert bool((safe >= 0).all() and (safe <= 1).all())
    assert float(scale.min()) < 1.0
    assert float(clip.max()) > 0.0
    print("V2.40 gamut tests: PASS")


if __name__ == "__main__":
    main()
