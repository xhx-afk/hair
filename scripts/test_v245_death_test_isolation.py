from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.appearance_probe_v245 import AppearanceProbeV245
from models.matte_probe_v245 import MatteProbeV245
from models.occluder_probe_v245 import OccluderProbeV245


def test_isolation_contract():
    base = torch.rand(1, 3, 16, 16)
    carrier = torch.rand_like(base)
    coarse = torch.zeros(1, 1, 16, 16); coarse[:, :, 4:12, 4:12] = 1
    alpha = coarse.clone()
    matte_hair = carrier.clone()
    assert float((matte_hair - carrier).abs().max()) < 1e-7
    trusted = AppearanceProbeV245().trusted_core(coarse, torch.ones_like(coarse))
    assert float((trusted - AppearanceProbeV245().trusted_core(coarse, torch.ones_like(coarse))).abs().max()) < 1e-7
    _, protected = OccluderProbeV245.apply(base_rgb=base, hair_rgb=carrier, hair_alpha=alpha, occluder_alpha=torch.zeros_like(alpha))
    assert float((carrier - carrier).abs().max()) < 1e-7
    assert float((alpha - alpha).abs().max()) < 1e-7
    assert protected.shape == base.shape


if __name__ == "__main__":
    test_isolation_contract(); print("V2.45 isolation test: PASS")
