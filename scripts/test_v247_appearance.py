"""Ten deterministic V2.47 carrier-first checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.appearance_probe_v247 import AppearanceProbeV247
from models.carrier_reference_error_v847 import CarrierReferenceErrorEstimatorV847
from models.hair_photometric_residual_v847 import HairPhotometricResidualV847
from utils.v247_appearance_metrics import classify_components


def main() -> None:
    size = 64; base = torch.full((1, 3, size, size), .45); reference = torch.full_like(base, .46); carrier = base.clone(); anchor = carrier.clone(); target = torch.zeros(1, 1, size, size); target[:, :, 8:56, 8:56] = 1; ref_mask = target.clone()
    estimator = CarrierReferenceErrorEstimatorV847(); perfect = estimator(carrier_rgb=carrier, reference_rgb=reference, carrier_hair_mask=target, reference_hair_mask=ref_mask)
    assert float(perfect["l_gate_strength"].max()) <= .05; assert float(perfect["ab_gate_strength"].max()) <= .05
    bad_l = estimator(carrier_rgb=carrier, reference_rgb=torch.full_like(base, .75), carrier_hair_mask=target, reference_hair_mask=ref_mask); assert float(bad_l["l_gate_strength"].max()) > .05 and float(bad_l["ab_gate_strength"].max()) < .05
    bad_ab = estimator(carrier_rgb=carrier, reference_rgb=torch.tensor([[[[.15]], [[.55]], [[.15]]]]).expand_as(base), carrier_hair_mask=target, reference_hair_mask=ref_mask); assert float(bad_ab["ab_gate_strength"].max()) > .05
    high = estimator(carrier_rgb=carrier, reference_rgb=torch.tensor([[[[.8]], [[.1]], [[.1]]]]).expand_as(base), carrier_hair_mask=target, reference_hair_mask=ref_mask); assert float(high["ab_gate_strength"].max()) <= 1.0
    residual = HairPhotometricResidualV847(); output, aux = residual(carrier_rgb=carrier, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=ref_mask, return_aux=True); assert float(aux["delta_ab_center"].norm().max()) <= 6.0 + 1e-5
    outside = (output - carrier).abs() * (1 - aux["target_hair_soft"]); assert float(outside.max()) < 1e-5
    assert float(aux["shadow_chroma_scale"].min()) >= .75 - 1e-5
    assert torch.allclose(aux["plausibility_scale"], torch.ones_like(aux["plausibility_scale"]))
    probe = AppearanceProbeV247(); outputs, variants = probe(carrier_rgb=carrier, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=ref_mask, strong_anchor_rgb=anchor, trusted_alpha=target, return_aux=True); assert set(variants) == {"c0", "c1", "c2", "c3", "c4", "c5"}; assert torch.equal(outputs["c0_rgb"], carrier)
    rows = [{"c0_appearance_metric_valid": 1, "c5_carrier_mid_structure_corr": .99, "c5_carrier_gradient_structure_corr": .99, "c5_carrier_relative_mid_energy": 1., "c5_carrier_relative_hf_energy": 1., "c5_non_hair_rgb_change": 0., "c1_l_q50_error": 1., "c0_l_q50_error": 3., "c1_median_ab_error": 1., "c0_median_ab_error": 1., "c2_median_ab_error": .8, "c2_chroma_error": .8, "c0_chroma_error": 2., "c2_stable_hue_error_deg": 1., "c0_stable_hue_error_deg": 1., "c3_median_ab_error": .8, "c1_chroma_error": 1., "c4_shadow_to_midtone_chroma_ratio": .8, "c3_shadow_to_midtone_chroma_ratio": 1., "c4_highlight_to_midtone_chroma_ratio": .9, "c3_highlight_to_midtone_chroma_ratio": 1., "c4_median_ab_error": .8, "c5_chroma_error": .8, "c4_chroma_error": .8}]
    assert "l_module" in classify_components(rows)
    print("V2.47 appearance tests: PASS (10 checks)")


if __name__ == "__main__": main()
