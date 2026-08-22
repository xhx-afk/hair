"""Combined contact/contamination risk must suppress uncertain hair."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.face_overlap_contamination_guard_v841 import FaceOverlapContaminationGuardV841


def main() -> None:
    shape = (1, 1, 32, 32)
    alpha = torch.ones(shape); rgb = torch.full((1, 3, 32, 32), 0.4)
    guarded, aux = FaceOverlapContaminationGuardV841()(alpha=alpha, allowed_hair=alpha,
        coarse_target_hair_mask=alpha, source_face_mask=alpha, source_skin_mask=alpha,
        strong_anchor_rgb=rgb, base_rgb=rgb, new_hair_rgb=rgb,
        face_contact_ring=alpha, face_intrusion_risk=alpha, hair_core=torch.zeros_like(alpha),
        anchor_hair_evidence=torch.full_like(alpha, 0.10), return_aux=True)
    assert float(aux["combined_contamination_risk"].mean()) > 0.65
    assert float(guarded.mean()) < 0.30
    print("V2.41.2 final guard tests: PASS")


if __name__ == "__main__":
    main()
