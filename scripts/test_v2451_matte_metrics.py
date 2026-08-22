from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v245_matte_metrics import matte_metric_tensors


def main() -> None:
    shape = (1, 1, 16, 16); target = torch.ones(shape); source = torch.zeros(shape); face = torch.ones(shape); zero = torch.zeros(shape); rgb = torch.rand(1, 3, 16, 16)
    metrics = matte_metric_tensors(base_rgb=rgb, carrier_rgb=rgb, alpha=target, coarse_target_hair_mask=target, source_hair_mask=source, source_face_mask=face, source_skin_mask=zero, hair_support=target, independent_candidate=zero)
    assert torch.allclose(metrics["new_hair_coverage"], metrics["target_hair_coverage"])
    assert "alpha_edge_width_10_90" not in metrics
    print("V2.45.1 matte metrics test: PASS")


if __name__ == "__main__": main()
