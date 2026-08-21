"""Synthetic contracts for the V2.38 high-resolution matte refiner."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.target_hair_matte_refiner_v838 import TargetHairMatteRefinerV838


def _run(face_guard: float):
    h = w = 64
    base = torch.full((1, 3, h, w), 0.30)
    new_hair = torch.full_like(base, 0.70)
    anchor = base.clone()
    anchor[..., 16:48, 16:48] = 0.75
    allowed = torch.zeros(1, 1, h, w)
    allowed[..., 16:48, 16:48] = 1.0
    face = torch.zeros_like(allowed)
    face[..., 12:32, 12:52] = 1.0
    ring = face * ((allowed - (-torch.nn.functional.max_pool2d(-allowed, 11, 1, 5))).clamp(0, 1))
    risk = ring * 0.8
    refiner = TargetHairMatteRefinerV838(ring_radius=5, face_guard=face_guard)
    return refiner(
        base_rgb=base, strong_anchor_rgb=anchor, new_hair_rgb=new_hair,
        coarse_target_hair_mask=allowed, occlusion_allowed_hair=allowed,
        source_face_mask=face, source_skin_mask=face,
        face_contact_ring=ring, face_intrusion_risk=risk, return_aux=True,
    )


def main() -> None:
    alpha_soft, aux_soft = _run(0.35)
    alpha_open, _ = _run(0.0)
    assert float(alpha_soft[..., 24:40, 24:40].mean()) > 0.95
    assert float(alpha_soft[..., :8, :8].max()) == 0.0
    assert float(aux_soft["target_hair_transition"].sum()) > 0.0
    unique = torch.unique(alpha_soft)
    assert unique.numel() > 3, "matte collapsed to a binary nearest edge"
    # Guarding contact pixels must not erase the interior core.
    assert float((alpha_open[..., 24:40, 24:40] - alpha_soft[..., 24:40, 24:40]).abs().max()) < 1e-6
    assert float(alpha_soft[..., 16:24, 24:40].mean()) < float(alpha_open[..., 16:24, 24:40].mean())
    print("V2.38 matte refiner tests: PASS")


if __name__ == "__main__":
    main()
