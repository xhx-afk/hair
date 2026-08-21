"""Synthetic distribution metric contracts for V2.39."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v239_metrics import v239_metric_tensors


def main() -> None:
    h = w = 32
    base = torch.full((1, 3, h, w), 0.25)
    reference = torch.full_like(base, 0.20)
    reference[..., 8:24, 8:24] = torch.tensor([0.12, 0.055, 0.025]).view(1, 3, 1, 1)
    hair = torch.zeros(1, 1, h, w)
    hair[..., 8:24, 8:24] = 1.0
    metrics = v239_metric_tensors(
        base_rgb=base, final_rgb=reference, color_reference_rgb=reference,
        coarse_target_hair_mask=hair, source_face_mask=torch.zeros_like(hair),
        hair_alpha_final=hair, allowed_hair_mask=hair, reference_hair_mask=hair,
    )
    assert float(metrics["hair_median_l_error"].item()) < 1e-4
    assert float(metrics["hair_l_q50_error"].item()) < 1e-4
    assert float(metrics["hair_hue_error_deg"].item()) < 1e-4
    assert float(metrics["low_chroma_tone_fidelity"].item()) > 0.99
    assert torch.isfinite(torch.stack(list(metrics.values()))).all()
    print("V2.39 distribution metrics tests: PASS")


if __name__ == "__main__":
    main()
