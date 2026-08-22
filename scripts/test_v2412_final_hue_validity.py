"""Final chromatic support is required for effective hue validity."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v241_metrics import v241_metric_tensors, aggregate_v2412_records


def main() -> None:
    shape = (1, 1, 8, 8)
    ref = torch.full((1, 3, 8, 8), 0.4); ref[:, 0] = 0.8; ref[:, 1] = 0.15
    final = torch.full_like(ref, 0.4)
    mask = torch.ones(shape); zeros = torch.zeros(shape)
    metrics = v241_metric_tensors(base_rgb=final, final_rgb=final, color_reference_rgb=ref,
        coarse_target_hair_mask=mask, source_face_mask=zeros, source_skin_mask=zeros,
        hair_alpha_final=mask, allowed_hair_mask=mask, reference_hair_mask=mask,
        strong_anchor_rgb=final, face_contact_ring=zeros, face_intrusion_risk=zeros,
        anchor_hair_evidence=mask, hue_metric_valid=torch.ones(1, 1, 1, 1))
    assert float(metrics["reference_hue_metric_valid"].mean()) == 1.0
    assert float(metrics["final_hue_support_valid"].mean()) == 0.0
    assert float(metrics["effective_hue_metric_valid"].mean()) == 0.0
    summary, failures = aggregate_v2412_records([{
        "sample_id": "gray", "stable_hue_error_deg": 0.0,
        "effective_hue_metric_valid": 0.0, "high_chroma_hue_metric_valid": 0.0,
        "highlight_hue_drift_valid": 0.0, "low_chroma_sample_valid": 1.0,
    }], min_samples=1)
    assert "V2412_HUE_INSUFFICIENT_VALID_SAMPLES" in failures
    assert summary["hue_valid_sample_count"] == 0
    print("V2.41.2 final hue validity tests: PASS")


if __name__ == "__main__":
    main()
