"""V2.37 metrics compare identity against Base, never Strong Anchor."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v237_metrics import v237_metric_tensors


def main() -> None:
    base = torch.zeros(1, 3, 16, 16)
    final = base.clone()
    anchor = torch.ones_like(base)
    reference = torch.full_like(base, 0.5)
    hair = torch.zeros(1, 1, 16, 16)
    hair[..., 4:12, 4:12] = 1
    face = torch.zeros_like(hair)
    alpha = hair.clone()
    metrics = v237_metric_tensors(
        base_rgb=base, final_rgb=final, color_reference_rgb=reference,
        target_hair_mask=hair, face_mask=face, hair_alpha=alpha,
    )
    assert torch.allclose(metrics["face_rgb_change_from_base"], torch.zeros(1))
    assert torch.allclose(metrics["non_hair_rgb_change_from_base"], torch.zeros(1))
    assert torch.isfinite(torch.stack(list(metrics.values()))).all()
    print("V2.37 metrics tests: PASS")


if __name__ == "__main__":
    main()
