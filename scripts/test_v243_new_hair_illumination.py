from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.hair_photometric_appearance_v843 import HairPhotometricAppearanceV843


def test_plausibility_is_soft():
    ref_ab = torch.tensor([[[[20.0]], [[10.0]]]])
    out, aux = HairPhotometricAppearanceV843()(reference_ab=ref_ab,
        final_l=torch.full((1, 1, 1, 1), 50.0), source_hair_l=torch.tensor([[[[10.0]]]]),
        reference_l=torch.tensor([[[[90.0]]]]), return_aux=True)
    assert 0.79 <= float(aux["plausibility_scale"]) <= 1.0
    assert out.shape == ref_ab.shape


if __name__ == "__main__":
    test_plausibility_is_soft(); print("V2.43 new-hair illumination test: PASS")
