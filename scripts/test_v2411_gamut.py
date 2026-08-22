"""V2.41.1 gamut largest-valid-scale and already-safe contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gamut_safe_lab_v841 import gamut_safe_lab_to_rgb_v841, lab_to_rgb_unclamped_v841


def main() -> None:
    unsafe = torch.tensor([[[[70.0]], [[70.0]], [[56.0]]]])
    raw = lab_to_rgb_unclamped_v841(unsafe)
    safe, aux = gamut_safe_lab_to_rgb_v841(unsafe, iterations=12, return_aux=True)
    assert bool(((raw < 0) | (raw > 1)).any())
    assert 0.5 < float(aux["gamut_scale_map"].item()) < 1.0
    assert bool(((safe >= 0) & (safe <= 1)).all())
    already_safe = torch.tensor([[[[55.0]], [[2.0]], [[1.0]]]])
    _, safe_aux = gamut_safe_lab_to_rgb_v841(already_safe, return_aux=True)
    assert float(safe_aux["gamut_scale_map"].item()) == 1.0
    print("V2.41.1 gamut tests: PASS")


if __name__ == "__main__":
    main()
