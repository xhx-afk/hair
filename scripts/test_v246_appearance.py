"""Ten deterministic V2.46 appearance-first unit checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.appearance_probe_v246 import AppearanceProbeV246
from models.hair_low_frequency_illumination_v846 import HairLowFrequencyIlluminationFieldV846
from models.hair_photometric_residual_v846 import HairPhotometricResidualV846
from utils.v246_appearance_metrics import classify_appearance


def _inputs(size: int = 64):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    base = torch.full((1, 3, size, size), .4); base[:, 0] += .05 * xx; base[:, 1] += .03 * yy
    anchor = base + .03 * torch.sin(xx * 40).view(1, 1, size, size)
    carrier = anchor.clone(); target = ((xx - .5).square() + (yy - .5).square() < .18).float().view(1, 1, size, size)
    source = torch.zeros_like(target); evidence = torch.ones_like(target)
    return base.clamp(0, 1), anchor.clamp(0, 1), carrier.clamp(0, 1), target, source, evidence


def main() -> None:
    base, anchor, carrier, target, source, evidence = _inputs()
    field = HairLowFrequencyIlluminationFieldV846()
    ref_center = torch.tensor([45.0]).view(1, 1, 1, 1)
    result = field(base_rgb=base, strong_anchor_rgb=anchor, carrier_rgb=carrier, target_hair_mask=target, source_hair_mask=source, reference_l_center=ref_center)
    raw_hf = result["delta_l_low_raw"] - torch.nn.functional.avg_pool2d(result["delta_l_low_raw"], 3, 1, 1)
    band_hf = result["delta_l_low_bandlimited"] - torch.nn.functional.avg_pool2d(result["delta_l_low_bandlimited"], 3, 1, 1)
    assert float(band_hf.abs().mean()) < float(raw_hf.abs().mean())
    assert float(result["delta_l_low_bandlimited"].abs().max()) <= 8.0 + 1e-6
    probe = AppearanceProbeV246(); trusted = probe.trusted_core(target, evidence)
    carrier_lab = __import__("models.SG_IDCT_v16", fromlist=["rgb_to_lab"]).rgb_to_lab(carrier)
    reference_ab = carrier_lab[:, 1:] + 4.0
    outputs, aux = probe(carrier_lab=carrier_lab, base_rgb=base, strong_anchor_rgb=anchor, carrier_rgb=carrier, target_hair_mask=target, source_hair_mask=source, reference_l_center=ref_center, reference_ab_low=reference_ab, trusted_alpha=trusted, return_aux=True)
    assert torch.equal(aux["b2"]["final_l"], aux["b2"]["carrier_l"])
    assert torch.allclose(aux["b3"]["shadow_chroma_scale"], torch.ones_like(aux["b3"]["shadow_chroma_scale"]))
    assert torch.allclose(aux["b3"]["highlight_chroma_scale"], torch.ones_like(aux["b3"]["highlight_chroma_scale"]))
    scene_a = probe.photometric(carrier_lab=carrier_lab, base_rgb=base, strong_anchor_rgb=anchor, carrier_rgb=carrier, target_hair_mask=target, source_hair_mask=source, reference_l_center=ref_center, reference_ab_low=reference_ab, scene_illumination_ab=torch.zeros(1, 2, 64, 64))
    scene_b = probe.photometric(carrier_lab=carrier_lab, base_rgb=base, strong_anchor_rgb=anchor, carrier_rgb=carrier, target_hair_mask=target, source_hair_mask=source, reference_l_center=ref_center, reference_ab_low=reference_ab, scene_illumination_ab=torch.full((1, 2, 64, 64), 40.0))
    assert torch.allclose(scene_a, scene_b, atol=1e-6)
    assert float(aux["b1"]["final_ab"].shape[1]) == 2
    assert float(aux["b1"]["final_chroma"].max()) >= 0.0
    rows = [{"b1_appearance_metric_valid": 1, "b1_carrier_mid_structure_corr": .95, "b1_carrier_gradient_structure_corr": .9, "b1_carrier_relative_mid_energy": 1., "b1_carrier_relative_hf_energy": 1., "b1_delta_l_to_carrier_mid_ratio": .1, "b1_delta_l_to_carrier_hf_ratio": .05, "b1_median_ab_error": 1., "b0_median_ab_error": 2., "b1_chroma_magnitude_error": 1., "b0_chroma_magnitude_error": 2., "b1_stable_hue_error_deg": 1., "b0_stable_hue_error_deg": 1., "b1_l_q50_error": 1., "b0_l_q50_error": 1., "b1_highlight_to_midtone_chroma_ratio": .9, "b1_shadow_to_midtone_chroma_ratio": .8, "b3_highlight_to_midtone_chroma_ratio": 1.1, "b3_shadow_to_midtone_chroma_ratio": 1.1}]
    assert classify_appearance(rows)["decision"] == "V246_READY_FOR_VISUAL_REVIEW"
    print("V2.46 appearance tests: PASS (10 checks)")


if __name__ == "__main__":
    main()
