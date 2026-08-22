from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v245_appearance_metrics import classify_appearance


def main() -> None:
    rows = [{"a1_appearance_metric_valid": 1, "a1_median_ab_error": 2, "a0_median_ab_error": 5, "a2_stable_hue_error_deg": 1, "a1_stable_hue_error_deg": 6, "a2_reference_chroma_error": 1, "a1_reference_chroma_error": 3, "a1_carrier_mid_structure_corr": .9, "a2_carrier_mid_structure_corr": .9, "a1_carrier_gradient_structure_corr": .9, "a1_intra_output_highlight_to_midtone_chroma_ratio": .9, "a3_intra_output_highlight_to_midtone_chroma_ratio": 1.2, "a1_intra_output_shadow_to_midtone_chroma_ratio": .9, "a3_intra_output_shadow_to_midtone_chroma_ratio": 1.2, "a3_median_ab_error": 3}]
    result = classify_appearance(rows); assert result["scene_tint"] == "GLOBAL_SCENE_TINT_HARMFUL"; assert result["photometric_overall"] == "PHOTOMETRIC_HAS_VALUE"; print("V2.45.1 appearance metrics test: PASS")


if __name__ == "__main__": main()
