"""Synthetic contracts for the V2.38 target-hair occlusion resolver."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.target_hair_occlusion_resolver_v838 import TargetHairOcclusionResolverV838


def main() -> None:
    torch.manual_seed(238)
    h = w = 64
    base = torch.full((1, 3, h, w), 0.35)
    anchor = base.clone()
    # High-frequency evidence is present throughout the target hair region.
    stripe = (torch.arange(w).view(1, 1, 1, w) % 4 < 2).float()
    anchor = (anchor + 0.22 * stripe).clamp(0, 1)
    coarse = torch.zeros(1, 1, h, w)
    coarse[..., 14:50, 14:50] = 1.0
    face = torch.zeros_like(coarse)
    face[..., 24:56, 24:56] = 1.0
    skin = face.clone()

    resolver = TargetHairOcclusionResolverV838(contact_radius=4, risk_strength=0.65)
    allowed, aux = resolver(
        coarse_target_hair_mask=coarse, source_face_mask=face,
        source_skin_mask=skin, strong_anchor_rgb=anchor, base_rgb=base,
        return_aux=True,
    )
    overlap = coarse * face
    assert float(allowed[overlap > 0].mean()) > 0.20, "face overlap was hard-vetoed"
    assert float(allowed[coarse <= 0].max()) == 0.0, "hair ownership escaped topology"
    assert float(aux["face_intrusion_risk"].max()) <= 1.0

    # With no overlap the resolver is topology-preserving exactly.
    no_face = torch.zeros_like(face)
    allowed_no_face = resolver(
        coarse_target_hair_mask=coarse, source_face_mask=no_face,
        source_skin_mask=no_face, strong_anchor_rgb=anchor, base_rgb=base,
    )
    assert torch.allclose(allowed_no_face, coarse, atol=1e-6)
    print("V2.38 occlusion resolver tests: PASS")


if __name__ == "__main__":
    main()
