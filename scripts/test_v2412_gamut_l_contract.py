"""Gamut conversion exposes and uses the safe L channel consistently."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841


def main() -> None:
    lab = torch.zeros(1, 3, 4, 4); lab[:, 0] = -20.0; lab[:, 1] = 40.0; lab[:, 2] = -20.0
    rgb, aux = gamut_safe_lab_to_rgb_v841(lab, return_aux=True)
    assert torch.isfinite(rgb).all()
    assert torch.allclose(aux["gamut_safe_l"], aux["lab_safe_l"])
    assert float(aux["lab_input_l"].min()) < 0.0
    assert float(aux["gamut_safe_l"].min()) >= 1.0
    assert torch.allclose(aux["gamut_safe_lab"][:, :1], aux["gamut_safe_l"])
    print("V2.41.2 gamut L contract tests: PASS")


if __name__ == "__main__":
    main()
