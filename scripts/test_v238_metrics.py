"""Synthetic contracts for V2.38 Base-relative diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v238_metrics import v238_metric_tensors


def main() -> None:
    h = w = 32
    base = torch.full((1, 3, h, w), 0.25)
    reference = base.clone()
    reference[..., 8:24, 8:24] = torch.tensor([0.80, 0.15, 0.35]).view(1, 3, 1, 1)
    hair = torch.zeros(1, 1, h, w)
    hair[..., 8:24, 8:24] = 1.0
    face = torch.zeros_like(hair)
    alpha = hair.clone()
    halfway = base + 0.5 * (reference - base) * hair
    metrics = v238_metric_tensors(
        base_rgb=base, final_rgb=halfway, color_reference_rgb=reference,
        coarse_target_hair_mask=hair, source_face_mask=face,
        hair_alpha_final=alpha, allowed_hair_mask=hair,
    )
    assert float(metrics["hair_reference_progress"].item()) > 0.99
    assert float(metrics["hair_core_full_transfer_fraction"].item()) < 0.01
    assert float(metrics["base_leakage_fraction"].item()) < 0.01
    assert float(metrics["hair_undertransfer_fraction"].item()) < 0.01

    unchanged = v238_metric_tensors(
        base_rgb=base, final_rgb=base, color_reference_rgb=reference,
        coarse_target_hair_mask=hair, source_face_mask=face,
        hair_alpha_final=alpha, allowed_hair_mask=hair,
    )
    assert float(unchanged["hair_undertransfer_fraction"].item()) > 0.99
    assert float(unchanged["hair_low_transfer_fraction"].item()) > 0.99
    print("V2.38 metrics tests: PASS")


if __name__ == "__main__":
    main()
