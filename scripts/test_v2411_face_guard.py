"""Unified evidence guard contracts: invalid, uncertain, and legal bang."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.face_overlap_contamination_guard_v841 import FaceOverlapContaminationGuardV841


def run(evidence: float, ring: float = 0.0):
    shape = (1, 1, 32, 32)
    alpha = torch.ones(shape); allowed = torch.ones(shape); coarse = torch.ones(shape)
    face = torch.zeros(shape); face[..., 8:24, 8:24] = 1
    skin = face.clone(); base = torch.full((1, 3, 32, 32), 0.7)
    anchor = base.clone(); new_hair = torch.full_like(base, 0.05)
    return FaceOverlapContaminationGuardV841()(
        alpha=alpha, allowed_hair=allowed, coarse_target_hair_mask=coarse,
        source_face_mask=face, source_skin_mask=skin, strong_anchor_rgb=anchor,
        base_rgb=base, new_hair_rgb=new_hair,
        face_contact_ring=face * ring, face_intrusion_risk=face * 0.9,
        hair_core=torch.ones_like(alpha), anchor_hair_evidence=torch.full_like(alpha, evidence),
        return_aux=True,
    )


def main() -> None:
    invalid, invalid_aux = run(0.10)
    assert float(invalid_aux["anchor_hair_evidence"].min()) < 0.25
    assert float(invalid[..., 12:20, 12:20].max()) <= 0.051
    uncertain, uncertain_aux = run(0.35)
    assert float(uncertain_aux["deep_uncertain"].max()) > 0
    assert float(uncertain[..., 12:20, 12:20].mean()) < 0.30
    legal, legal_aux = run(0.80)
    assert float(legal_aux["anchor_hair_evidence"].max()) > 0.65
    assert float(legal[..., 12:20, 12:20].mean()) > 0.60
    contact, contact_aux = run(0.10, ring=1.0)
    assert float(contact_aux["face_contact_contamination_risk"].max()) > 0
    assert float(contact[..., 12:20, 12:20].mean()) < 1.0
    print("V2.41.1 face guard tests: PASS")


if __name__ == "__main__":
    main()
