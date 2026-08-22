"""Synthetic V2.41 metrics are finite and include new fields."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v241_metrics import v241_metric_tensors


def main() -> None:
    shape = (2, 3, 16, 16)
    rgb = torch.full(shape, 0.35)
    mask = torch.ones(2, 1, 16, 16)
    zero = torch.zeros_like(mask)
    metrics = v241_metric_tensors(
        base_rgb=rgb, final_rgb=rgb, color_reference_rgb=rgb,
        coarse_target_hair_mask=mask, source_face_mask=zero, source_skin_mask=zero,
        hair_alpha_final=mask, allowed_hair_mask=mask, reference_hair_mask=mask,
        strong_anchor_rgb=rgb, face_contact_ring=zero, face_intrusion_risk=zero,
    )
    required = ("hair_hue_spatial_std", "hair_hue_outlier_fraction", "highlight_hue_drift",
                "gamut_unsafe_fraction", "median_gamut_scale", "face_overlap_bleed",
                "deep_skin_overlap_bleed", "uncertain_skin_overlap_bleed")
    assert all(name in metrics for name in required)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    print("V2.41 metrics tests: PASS")


if __name__ == "__main__":
    main()
