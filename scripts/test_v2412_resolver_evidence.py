"""V2.41.2 resolver must use independent hair evidence in face overlap."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import inspect

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.target_hair_occlusion_resolver_v8412 import TargetHairOcclusionResolverV8412


def main() -> None:
    shape = (1, 1, 32, 32)
    coarse = torch.ones(shape); face = torch.ones(shape); skin = torch.ones(shape)
    rgb = torch.full((1, 3, 32, 32), 0.35)
    resolver = TargetHairOcclusionResolverV8412()
    assert "risk_strength" not in inspect.signature(TargetHairOcclusionResolverV8412).parameters
    low, low_aux = resolver(coarse_target_hair_mask=coarse, source_face_mask=face,
        source_skin_mask=skin, strong_anchor_rgb=rgb, base_rgb=rgb,
        anchor_hair_evidence=torch.full(shape, 0.10), return_aux=True)
    high, high_aux = resolver(coarse_target_hair_mask=coarse, source_face_mask=face,
        source_skin_mask=skin, strong_anchor_rgb=rgb, base_rgb=rgb,
        anchor_hair_evidence=torch.full(shape, 0.90), return_aux=True)
    assert float(high.mean()) > float(low.mean()) + 0.10
    assert float(low_aux["topology_connectedness"][..., 8:24, 8:24].mean()) > 0.9
    assert float(low_aux["overlap_confidence"].mean()) < 0.3
    print("V2.41.2 resolver evidence tests: PASS")


if __name__ == "__main__":
    main()
