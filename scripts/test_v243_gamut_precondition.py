from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.gamut_precondition_v843 import gamut_precondition_v843
from models.gamut_safe_lab_v841 import lab_to_rgb_unclamped_v841


def test_gamut_precondition_reduces_unsafe_chroma():
    lab = torch.tensor([[[[60.0]], [[120.0]], [[100.0]]]])
    conditioned, scale = gamut_precondition_v843(lab)
    assert float(scale.max()) < 1.0
    rgb = lab_to_rgb_unclamped_v841(conditioned)
    assert bool(torch.isfinite(rgb).all())


if __name__ == "__main__":
    test_gamut_precondition_reduces_unsafe_chroma(); print("V2.43 gamut precondition test: PASS")
