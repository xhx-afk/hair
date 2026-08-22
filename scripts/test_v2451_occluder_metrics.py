from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v245_occluder_metrics import occluder_metric_tensors


def main() -> None:
    base = torch.zeros(1, 3, 20, 20); current = base.clone(); protected = base.clone(); mask = torch.zeros(1, 1, 20, 20); mask[:, :, :4, :] = 1; current[:, :, :4, :] = .1
    metrics = occluder_metric_tensors(base_rgb=base, current_rgb=current, protected_rgb=protected, occluder_alpha=mask, hair_alpha=mask, real_mask=mask, synthetic_mask=torch.zeros_like(mask))
    assert abs(float(metrics["current_occluder_p90"][0]) - .1) < 1e-5; assert float(metrics["real_occluder_pixel_count"][0]) == 80
    print("V2.45.1 occluder metrics test: PASS")


if __name__ == "__main__": main()
