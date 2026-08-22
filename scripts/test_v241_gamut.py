"""Synthetic V2.41 gamut-search contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841, lab_to_rgb_unclamped_v841


def main() -> None:
    lab = torch.zeros(1, 3, 8, 8)
    lab[:, 0], lab[:, 1], lab[:, 2] = 70.0, 105.0, 85.0
    raw = lab_to_rgb_unclamped_v841(lab)
    safe, aux = gamut_safe_lab_to_rgb_v841(lab, return_aux=True)
    assert bool(((raw < 0) | (raw > 1)).any())
    assert bool(((safe >= 0) & (safe <= 1)).all())
    assert float(aux["gamut_scale_map"].min()) < 1.0
    assert float(aux["gamut_unsafe_fraction"].max()) > 0.0
    print("V2.41 gamut tests: PASS")


if __name__ == "__main__":
    main()
