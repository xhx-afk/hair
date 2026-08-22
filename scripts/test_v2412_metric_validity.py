"""Invalid hue samples must not become zero-error aggregate samples."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v241_metrics import aggregate_v2412_records


def main() -> None:
    records = [
        {"sample_id": "a", "stable_hue_error_deg": 20.0, "hue_metric_valid": 1.0,
         "effective_hue_metric_valid": 1.0, "high_chroma_hue_metric_valid": 1.0,
         "highlight_hue_drift": 12.0, "highlight_hue_metric_valid": 1.0,
         "highlight_hue_drift_valid": 1.0, "low_chroma_sample_valid": 1.0,
         "low_chroma_hair_l_error": 1.0, "low_chroma_tone_fidelity": 0.8},
        {"sample_id": "b", "stable_hue_error_deg": 10.0, "hue_metric_valid": 1.0,
         "effective_hue_metric_valid": 1.0, "high_chroma_hue_metric_valid": 1.0,
         "highlight_hue_drift": 8.0, "highlight_hue_metric_valid": 1.0,
         "highlight_hue_drift_valid": 1.0, "low_chroma_sample_valid": 1.0,
         "low_chroma_hair_l_error": 2.0, "low_chroma_tone_fidelity": 0.7},
        {"sample_id": "c", "stable_hue_error_deg": 0.0, "hue_metric_valid": 0.0,
         "effective_hue_metric_valid": 0.0, "high_chroma_hue_metric_valid": 0.0,
         "highlight_hue_drift": 0.0, "highlight_hue_metric_valid": 0.0,
         "highlight_hue_drift_valid": 0.0, "low_chroma_sample_valid": 0.0},
        {"sample_id": "d", "stable_hue_error_deg": 0.0, "hue_metric_valid": 0.0,
         "effective_hue_metric_valid": 0.0, "high_chroma_hue_metric_valid": 0.0,
         "highlight_hue_drift": 0.0, "highlight_hue_metric_valid": 0.0,
         "highlight_hue_drift_valid": 0.0, "low_chroma_sample_valid": 0.0},
    ]
    summary, failures = aggregate_v2412_records(records, min_samples=2)
    assert summary["median_stable_hue_error_deg"] == 15.0
    assert summary["hue_valid_sample_count"] == 2
    assert not failures
    print("V2.41.2 metric validity tests: PASS")


if __name__ == "__main__":
    main()
