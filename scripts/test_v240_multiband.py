"""Synthetic contracts for luminance/chroma texture preservation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_local_recomposition_v840 import HairLocalRecompositionV840
from models.v840_runtime_inputs import build_v840_runtime_inputs
from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def main() -> None:
    torch.manual_seed(240)
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    stripes = (((xx * 3 + yy) % 7) < 3).float()
    anchor = torch.full((1, 3, h, w), 0.35)
    anchor[:, 0] += 0.24 * stripes
    anchor[:, 1] += 0.10 * stripes
    anchor[:, 2] += 0.05 * stripes
    base = torch.full_like(anchor, 0.40)
    reference = torch.full_like(anchor, 0.12)
    reference[:, 0] = 0.34
    reference[:, 1] = 0.16
    reference[:, 2] = 0.08
    mask = torch.zeros(1, 1, h, w)
    mask[..., 8:56, 8:56] = 1
    runtime = build_v840_runtime_inputs(
        base_rgb=base, strong_anchor_rgb=anchor, color_reference_rgb=reference,
        target_illumination_rgb=base, coarse_target_hair_mask=mask,
        source_face_mask=torch.zeros_like(mask), source_skin_mask=torch.zeros_like(mask),
        reference_hair_mask=mask,
    )
    final, aux = HairLocalRecompositionV840()(return_aux=True, **runtime)
    anchor_l, final_l = rgb_to_lab(anchor)[:, :1], rgb_to_lab(final)[:, :1]
    mid_anchor = gaussian_blur_v829(anchor_l, 3) - gaussian_blur_v829(anchor_l, 11)
    mid_final = gaussian_blur_v829(final_l, 3) - gaussian_blur_v829(final_l, 11)
    core = torch.zeros_like(aux["target_hair_alpha_final"])
    core[..., 20:44, 20:44] = 1.0
    ratio = (mid_final.abs() * core).sum() / (mid_anchor.abs() * core).sum().clamp_min(1e-5)
    assert 0.55 < float(ratio) < 1.45, float(ratio)
    assert float(aux["carrier_l_mid"].abs().mean()) > 0.01
    assert float(aux["face_contact_contamination_risk"].max()) == 0.0
    assert torch.isfinite(final).all()
    print("V2.40 multiband recolor tests: PASS")


if __name__ == "__main__":
    main()
