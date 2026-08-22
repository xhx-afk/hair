"""Low-chroma classification is sample-level, not pixel-subset based."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v241_metrics import aggregate_v2412_records, v241_metric_tensors


def metric(reference: torch.Tensor):
    shape = (1, 1, 8, 8); mask = torch.ones(shape); zeros = torch.zeros(shape)
    return v241_metric_tensors(base_rgb=reference, final_rgb=reference, color_reference_rgb=reference,
        coarse_target_hair_mask=mask, source_face_mask=zeros, source_skin_mask=zeros,
        hair_alpha_final=mask, allowed_hair_mask=mask, reference_hair_mask=mask,
        strong_anchor_rgb=reference, face_contact_ring=zeros, face_intrusion_risk=zeros,
        anchor_hair_evidence=mask, hue_metric_valid=torch.ones(1, 1, 1, 1))


def main() -> None:
    colorful = torch.full((1, 3, 8, 8), 0.40)
    colorful[:, 0] = 0.75; colorful[:, 1] = 0.18
    colorful[..., 0, 0] = 0.40
    gray = torch.full((1, 3, 8, 8), 0.40)
    assert float(metric(colorful)["low_chroma_sample_valid"].mean()) == 0.0
    assert float(metric(gray)["low_chroma_sample_valid"].mean()) == 1.0
    rows = [{"sample_id": str(i), "low_chroma_sample_valid": 1.0,
             "low_chroma_hair_l_error": float(i), "low_chroma_tone_fidelity": 1.0}
            for i in range(2)] + [{"sample_id": str(i), "low_chroma_sample_valid": 0.0,
             "low_chroma_hair_l_error": 100.0, "low_chroma_tone_fidelity": 0.0}
            for i in range(2, 10)]
    summary, _ = aggregate_v2412_records(rows, min_samples=2)
    assert summary["low_chroma_valid_sample_count"] == 2
    assert summary["median_low_chroma_hair_l_error"] == 0.5
    print("V2.41.2 low-chroma validity tests: PASS")


if __name__ == "__main__":
    main()
