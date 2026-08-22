"""V2.41.1 stable hue gray-selection and ash-brown contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import lab_to_rgb
from models.reference_stable_hue_palette_v841 import ReferenceStableHuePaletteV841
from models.target_stable_hue_chroma_field_v841 import TargetStableHueChromaFieldV841


def main() -> None:
    h = w = 20
    l = torch.full((1, 1, h, w), 55.0)
    # 90% near-gray pixels and 10% colored outliers: global chroma must stay low.
    ab_gray = torch.zeros(1, 2, h, w); ab_gray[:, 0] = 1.5; ab_gray[:, 1] = 0.5
    ab_gray[..., :2, :2] = torch.tensor([[[[8.0]], [[6.0]]]])
    reference = lab_to_rgb(torch.cat((l, ab_gray), dim=1)).clamp(0, 1)
    mask = torch.ones(1, 1, h, w)
    palette = ReferenceStableHuePaletteV841(min_support=4)(reference_rgb=reference, reference_hair_mask=mask)
    assert float(palette["global_chroma"].item()) < 4.0
    assert float(palette["valid_hue_fraction"].item()) < 0.2

    # Ash brown around C=8 should not receive the achromatic attenuation.
    ash_ab = torch.zeros_like(ab_gray); ash_ab[:, 0] = 6.4; ash_ab[:, 1] = 4.8
    ash = lab_to_rgb(torch.cat((l, ash_ab), dim=1)).clamp(0, 1)
    ash_palette = ReferenceStableHuePaletteV841(min_support=4)(reference_rgb=ash, reference_hair_mask=mask)
    field = TargetStableHueChromaFieldV841()(target_illumination_rgb=ash, target_hair_mask=mask, palette=ash_palette)
    assert float(ash_palette["global_chroma"].item()) > 5.0
    assert float(torch.linalg.vector_norm(field["target_ab"], dim=1).mean()) > 3.0
    print("V2.41.1 stable hue tests: PASS")


if __name__ == "__main__":
    main()
