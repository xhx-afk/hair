"""Synthetic contact guard test: color difference cannot erase confident hair."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.face_contact_guard_v840 import FaceContactContaminationGuardV840


def main() -> None:
    shape = (1, 1, 32, 32)
    alpha = torch.ones(shape)
    allowed = torch.ones(shape)
    face = torch.zeros(shape)
    face[..., 8:24, 8:24] = 1
    ring = face.clone()
    risk = ring * 0.9
    base = torch.full((1, 3, 32, 32), 0.75)
    anchor = base.clone()
    anchor[..., 8:24, 8:24] = 0.10
    new_hair = torch.full_like(base, 0.05)
    guarded, aux = FaceContactContaminationGuardV840()(
        alpha=alpha, allowed_hair=allowed, source_face_mask=face,
        source_skin_mask=face, face_contact_ring=ring, face_intrusion_risk=risk,
        strong_anchor_rgb=anchor, base_rgb=base, new_hair_rgb=new_hair,
        hair_core=torch.ones_like(alpha), return_aux=True,
    )
    assert float(guarded[..., 12:20, 12:20].mean()) > 0.9
    assert float(aux["face_contact_contamination_risk"].max()) < 1.0
    print("V2.40 face guard tests: PASS")


if __name__ == "__main__":
    main()
