"""Deep skin bleed is evaluated only for uncertain hair evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v241_metrics import v241_metric_tensors


def run(evidence: float):
    shape = (1, 1, 32, 32)
    base = torch.full((1, 3, 32, 32), 0.30)
    final = base + 0.20
    mask = torch.ones(shape)
    face = torch.ones(shape)
    ring = torch.zeros(shape)
    return v241_metric_tensors(base_rgb=base, final_rgb=final, color_reference_rgb=base,
        coarse_target_hair_mask=mask, source_face_mask=face, source_skin_mask=face,
        hair_alpha_final=mask, allowed_hair_mask=mask, reference_hair_mask=mask,
        strong_anchor_rgb=base, face_contact_ring=ring, face_intrusion_risk=ring,
        anchor_hair_evidence=torch.full_like(mask, evidence), hue_metric_valid=torch.ones(1, 1, 1, 1))


def main() -> None:
    clean, polluted = run(0.90), run(0.10)
    assert float(clean["deep_uncertain_skin_overlap_bleed"].mean()) < 1e-6
    assert float(polluted["deep_uncertain_skin_overlap_bleed"].mean()) > 0.1
    print("V2.41.2 deep skin metric tests: PASS")


if __name__ == "__main__":
    main()
