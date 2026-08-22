from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_appearance_v843 import photometric_chroma_scales_v843


def test_shadow_desaturation():
    scales = photometric_chroma_scales_v843(torch.tensor([[[[10.0]]]]))
    mid = photometric_chroma_scales_v843(torch.tensor([[[[50.0]]]]))
    assert float(scales["shadow_chroma_scale"]) < float(mid["shadow_chroma_scale"])


if __name__ == "__main__":
    test_shadow_desaturation(); print("V2.43 shadow photometric test: PASS")
