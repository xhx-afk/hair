from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.new_hair_illumination_v843 import new_hair_illumination_v843


def test_new_hair_does_not_follow_background_luminance():
    base = torch.full((1, 3, 32, 32), .2)
    anchor = torch.full_like(base, .5)
    target = torch.zeros(1, 1, 32, 32); target[:, :, 8:24, 8:24] = 1
    out = new_hair_illumination_v843(base_rgb=base, strong_anchor_rgb=anchor,
        target_hair_mask=target, source_hair_mask=torch.zeros_like(target),
        reference_l_center=torch.full((1, 1, 1, 1), 60.0))
    assert float(out["new_hair_l_low"][..., 12:20, 12:20].mean()) > 40.0


if __name__ == "__main__":
    test_new_hair_does_not_follow_background_luminance(); print("V2.43 new-hair ownership test: PASS")
