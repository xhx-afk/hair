"""Target illumination must affect low-frequency tone independently of Anchor."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_local_recomposition_v841 import HairLocalRecompositionV841
from models.v841_runtime_inputs import build_v841_runtime_inputs


def main() -> None:
    h = w = 48
    mask = torch.zeros(1, 1, h, w); mask[..., 8:40, 8:40] = 1
    anchor = torch.full((1, 3, h, w), 0.35)
    anchor[..., 15:33:2, 10:38] += 0.12
    base = torch.full_like(anchor, 0.38)
    reference = torch.full_like(anchor, 0.20); reference[:, 0] = 0.45
    target_dark = torch.full_like(anchor, 0.25)
    target_light = torch.full_like(anchor, 0.55)
    # Compare different target-local illumination structure rather than a
    # global offset, which must remain owned by the reference tone mapper.
    target_dark[..., 12:28, 12:28] += 0.08
    target_light[..., 20:36, 20:36] += 0.16
    common = dict(base_rgb=base, strong_anchor_rgb=anchor, color_reference_rgb=reference,
                 coarse_target_hair_mask=mask, source_face_mask=torch.zeros_like(mask),
                 source_skin_mask=torch.zeros_like(mask), reference_hair_mask=mask)
    model = HairLocalRecompositionV841()
    dark, dark_aux = model(return_aux=True, target_illumination_rgb=target_dark,
                            **common)
    light, light_aux = model(return_aux=True, target_illumination_rgb=target_light,
                             **common)
    assert torch.isfinite(dark).all() and torch.isfinite(light).all()
    assert float((light_aux["target_l_low"] - dark_aux["target_l_low"]).abs().mean()) > 1.0
    assert float((light - dark).abs().mean()) > 1e-3
    print("V2.41.1 illumination owner tests: PASS")


if __name__ == "__main__":
    main()
