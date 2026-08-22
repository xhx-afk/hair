"""V2.41.1 metrics use hair-core masks and expose heavy gamut gates."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v241_metrics import v241_metric_tensors


def main() -> None:
    rgb = torch.full((1, 3, 16, 16), 0.4)
    core = torch.zeros(1, 1, 16, 16); core[..., 4:12, 4:12] = 1
    all_hair = torch.ones_like(core); zero = torch.zeros_like(core)
    metrics = v241_metric_tensors(
        base_rgb=rgb, final_rgb=rgb, color_reference_rgb=rgb,
        coarse_target_hair_mask=all_hair, source_face_mask=zero, source_skin_mask=zero,
        hair_alpha_final=core, allowed_hair_mask=all_hair, reference_hair_mask=all_hair,
        strong_anchor_rgb=rgb, face_contact_ring=zero, face_intrusion_risk=zero,
        stable_hue_unit_ab=torch.tensor([[[[0.8]], [[0.6]]]]),
        hue_metric_valid=torch.ones(1, 1, 1, 1),
        gamut_aux={"gamut_unsafe_fraction": torch.tensor([0.1]),
                   "gamut_compressed_fraction": torch.tensor([0.1]),
                   "gamut_heavy_compression_fraction": torch.tensor([0.04]),
                   "median_gamut_scale": torch.tensor([0.6]),
                   "p10_gamut_scale": torch.tensor([0.55])},
    )
    for key in ("stable_hue_error_deg", "hair_hue_outlier_fraction", "highlight_hue_drift",
                "gamut_heavy_compression_fraction", "p10_gamut_scale"):
        assert key in metrics and torch.isfinite(metrics[key]).all()
    print("V2.41.1 metrics tests: PASS")


if __name__ == "__main__":
    main()
