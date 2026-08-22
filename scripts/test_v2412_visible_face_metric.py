"""Visible-face preservation excludes legitimate hair-covered pixels."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v241_metrics import v241_metric_tensors


def main() -> None:
    shape = (1, 1, 32, 32)
    base = torch.full((1, 3, 32, 32), 0.35)
    final = base.clone(); final[..., :16, :] = 0.80
    ref = torch.full_like(base, 0.35)
    hair = torch.ones(shape); face = torch.zeros(shape); face[..., :16, :] = 1
    alpha = hair.clone()
    zeros = torch.zeros_like(hair)
    metrics = v241_metric_tensors(base_rgb=base, final_rgb=final, color_reference_rgb=ref,
        coarse_target_hair_mask=hair, source_face_mask=face, source_skin_mask=face,
        hair_alpha_final=alpha, allowed_hair_mask=hair, reference_hair_mask=hair,
        strong_anchor_rgb=base, face_contact_ring=zeros, face_intrusion_risk=zeros,
        anchor_hair_evidence=torch.ones_like(hair), hue_metric_valid=torch.ones(1, 1, 1, 1))
    assert float(metrics["face_rgb_change_from_base"].mean()) > 0.1
    assert float(metrics["visible_face_rgb_change_from_base"].mean()) == 0.0
    print("V2.41.2 visible face metric tests: PASS")


if __name__ == "__main__":
    main()
