"""High-chroma hue validity has an independent pixel-support requirement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v241_metrics import v241_metric_tensors


def main() -> None:
    shape = (1, 1, 8, 8); mask = torch.ones(shape); zeros = torch.zeros(shape)
    rgb = torch.full((1, 3, 8, 8), 0.4); rgb[:, 0] = 0.43; rgb[:, 1] = 0.37
    metrics = v241_metric_tensors(base_rgb=rgb, final_rgb=rgb, color_reference_rgb=rgb,
        coarse_target_hair_mask=mask, source_face_mask=zeros, source_skin_mask=zeros,
        hair_alpha_final=mask, allowed_hair_mask=mask, reference_hair_mask=mask,
        strong_anchor_rgb=rgb, face_contact_ring=zeros, face_intrusion_risk=zeros,
        anchor_hair_evidence=mask, hue_metric_valid=torch.ones(1, 1, 1, 1))
    assert float(metrics["effective_hue_metric_valid"].mean()) == 1.0
    assert float(metrics["high_chroma_hue_metric_valid"].mean()) == 0.0
    print("V2.41.2 high-chroma hue validity tests: PASS")


if __name__ == "__main__":
    main()
