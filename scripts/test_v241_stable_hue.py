"""Synthetic stable-hue contract across dark/mid/highlight L."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import lab_to_rgb
from models.reference_stable_hue_palette_v841 import ReferenceStableHuePaletteV841
from models.target_stable_hue_chroma_field_v841 import TargetStableHueChromaFieldV841


def main() -> None:
    l = torch.linspace(25, 90, 64).view(1, 1, 8, 8)
    direction = torch.tensor([[[[0.8]], [[0.6]]]])
    ab = direction * torch.linspace(8, 30, 64).view(1, 1, 8, 8)
    reference = lab_to_rgb(torch.cat((l, ab), dim=1)).clamp(0, 1)
    mask = torch.ones(1, 1, 8, 8)
    palette = ReferenceStableHuePaletteV841(min_support=4)(reference_rgb=reference, reference_hair_mask=mask)
    stable = palette["stable_hue_unit_ab"]
    assert float(stable[:, 0].mean()) > 0.5 and float(stable[:, 1].mean()) > 0.3
    target = torch.full_like(reference, 0.4)
    field = TargetStableHueChromaFieldV841()(target_illumination_rgb=target, target_hair_mask=mask, palette=palette)
    field_angle = torch.atan2(field["target_ab"][:, 1], field["target_ab"][:, 0])
    assert float(field_angle.std()) < 1e-4
    print("V2.41 stable hue tests: PASS")


if __name__ == "__main__":
    main()
