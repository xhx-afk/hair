"""Synthetic overlap guard contract for uncertain skin and legal bangs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.face_overlap_contamination_guard_v841 import FaceOverlapContaminationGuardV841


def main() -> None:
    shape = (1, 1, 32, 32)
    alpha = torch.ones(shape)
    allowed = torch.ones(shape)
    coarse = torch.ones(shape)
    face = torch.zeros(shape); face[..., 8:24, 8:24] = 1
    skin = face.clone(); ring = face.clone(); risk = ring * 0.9
    base = torch.full((1, 3, 32, 32), 0.7)
    anchor = base.clone(); anchor[..., 8:24, 8:24] = 0.1
    new_hair = torch.full_like(base, 0.05)
    guarded, aux = FaceOverlapContaminationGuardV841()(
        alpha=alpha, allowed_hair=allowed, coarse_target_hair_mask=coarse,
        source_face_mask=face, source_skin_mask=skin, strong_anchor_rgb=anchor,
        base_rgb=base, new_hair_rgb=new_hair, face_contact_ring=ring,
        face_intrusion_risk=risk, hair_core=torch.ones_like(alpha),
        anchor_hair_evidence=torch.ones_like(alpha) * 0.8, return_aux=True,
    )
    assert torch.isfinite(guarded).all()
    assert float(aux["anchor_hair_evidence"][..., 12:20, 12:20].mean()) > 0.35
    assert float(guarded[..., 12:20, 12:20].mean()) > 0.05
    print("V2.41 face guard tests: PASS")


if __name__ == "__main__":
    main()
