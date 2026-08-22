"""Deterministic V2.47.1 correctness checks (diagnostic-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.appearance_probe_v247 import AppearanceProbeV247
from models.carrier_reference_error_v847 import CarrierReferenceErrorEstimatorV847
from models.hair_photometric_residual_v847 import HairPhotometricResidualV847, _masked_q, _region_ratio
from models.SG_IDCT_v16 import lab_to_rgb
from utils.v247_appearance_metrics import appearance_metric_tensors
from utils.v247_component_policy import build_component_policy, selected_components_for_group


def _scene(size: int = 64):
    base = torch.full((1, 3, size, size), .35)
    reference = torch.full_like(base, .62)
    target = torch.zeros((1, 1, size, size)); target[:, :, 8:56, 8:56] = 1
    return base, reference, target, target.clone()


def run_checks() -> None:
    base, reference, target, trusted = _scene()
    estimator = CarrierReferenceErrorEstimatorV847()

    # 1. L metrics must measure the current output, not the carrier baseline.
    metrics = appearance_metric_tensors(carrier_rgb=base, output_rgb=reference, trusted_core=trusted, reference_rgb=reference, reference_hair_mask=target, aux={})
    assert float(metrics["l_q50_error"].max()) < float(metrics["carrier_l_q50_error"].max())

    # 2. C5 uses a distinct plausibility-enabled residual instance.
    probe = AppearanceProbeV247()
    source_l = torch.full_like(target, .05); reference_l = torch.full_like(target, .95)
    outputs, aux = probe(carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, strong_anchor_rgb=base, source_hair_l=source_l, reference_l=reference_l, trusted_alpha=trusted, return_aux=True)
    assert torch.allclose(aux["c4"]["plausibility_scale"], torch.ones_like(aux["c4"]["plausibility_scale"]))
    assert float(aux["c5"]["plausibility_scale"].min()) < 1.0
    assert not torch.allclose(outputs["c4_rgb"], outputs["c5_rgb"])

    # 3. Trusted-core statistics are explicit; sparse supplied masks fall back.
    residual = HairPhotometricResidualV847()
    _, trusted_aux = residual(carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, carrier_stats_mask=trusted, return_aux=True)
    sparse = torch.zeros_like(target); sparse[:, :, 20:25, 20:25] = 1
    _, sparse_aux = residual(carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, carrier_stats_mask=sparse, return_aux=True)
    assert trusted_aux["stats_mask_source"] == "trusted_core" and float(trusted_aux["gate_metric_valid"].min()) == 1
    assert sparse_aux["stats_mask_source"] == "eroded_target"
    assert "gate_metric_valid" in sparse_aux and float(sparse_aux["gate_metric_valid"].max()) == 0

    # 4. Quantiles are hair masked and cannot move with background pixels.
    luma = torch.full((1, 1, 48, 48), 30.0); luma[:, :, :16] = 100.0
    hair = torch.ones_like(luma)
    assert float(_masked_q(luma, hair, .8)) < 60

    # 5. Highlight mean uses the highlight-region denominator.
    chroma = torch.full_like(luma, 5.0); chroma[:, :, :16, :48] = 20.0
    assert float(_region_ratio(chroma, luma, hair, .80)) > 15.0

    # 6. Shadow and highlight gates are independent fields.
    _, residual_aux = residual(carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, carrier_stats_mask=trusted, enable_shading=True, return_aux=True)
    assert "shadow_gate_strength" in residual_aux and "highlight_gate_strength" in residual_aux

    # 7. C0 is a true no-op in RGB and aux.
    assert torch.equal(outputs["c0_rgb"], base)
    for key in ("l_gate_strength", "ab_gate_strength", "shadow_gate_strength", "highlight_gate_strength", "shading_gate_strength", "plausibility_gate_strength", "l_delta", "gated_delta_l", "delta_ab_center"):
        assert float(aux["c0"][key].abs().max()) == 0.0
    assert float(aux["c0"]["total_gamut_scale"].min()) == 1.0 and float(aux["c0"]["no_op"].min()) == 1.0

    # 8. High chroma receives the conservative AB modifier.
    low_carrier = lab_to_rgb(torch.tensor([[[[50.0]], [[0.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    low_reference = lab_to_rgb(torch.tensor([[[[50.0]], [[8.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    high_carrier = lab_to_rgb(torch.tensor([[[[50.0]], [[30.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    high_reference = lab_to_rgb(torch.tensor([[[[50.0]], [[38.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    low = estimator(carrier_rgb=low_carrier, reference_rgb=low_reference, carrier_hair_mask=target, reference_hair_mask=target)
    high = estimator(carrier_rgb=high_carrier, reference_rgb=high_reference, carrier_hair_mask=target, reference_hair_mask=target)
    assert float(high["ab_gate_strength"].max()) <= float(low["ab_gate_strength"].max()) + 1e-5

    # 9. Active correction remains hair-only.
    changed = outputs["c3_rgb"]; strict = aux["c3"]["target_hair_soft"] < .01
    assert float(((changed - base).abs().mean(1, keepdim=True) * strict).max()) <= 1e-5

    def summary(decisions):
        return {"l_module": {"decision": decisions[0]}, "ab_module": {"decision": decisions[1]}, "shading_module": {"decision": decisions[2]}, "plausibility_module": {"decision": decisions[3]}}

    # 10. Two useful chroma groups enable a component and harmful group disables it.
    overall = summary(("L_MODULE_USEFUL", "AB_MODULE_USEFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL"))
    groups = {"low_chroma": summary(("L_MODULE_USEFUL", "AB_MODULE_USEFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL")), "normal_chroma": summary(("L_MODULE_USEFUL", "AB_MODULE_USEFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL")), "high_chroma": summary(("L_MODULE_USEFUL", "AB_MODULE_HARMFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL"))}
    policy = build_component_policy(overall, groups); assert policy["AB"]["global_enabled"] and "high_chroma" in policy["AB"]["disabled_groups"]

    # 11. One useful group plus two harmful groups is rejected globally.
    groups["normal_chroma"]["ab_module"]["decision"] = "AB_MODULE_HARMFUL"; policy_reject = build_component_policy(overall, groups)
    assert not policy_reject["AB"]["global_enabled"]

    # 12. Selected candidate flags are exactly the group policy flags.
    flags = selected_components_for_group(policy, "high_chroma")
    _, selected_aux = probe.run_selected(enable_l=flags["L"], enable_ab=flags["AB"], enable_shading=flags["Shading"], enable_plausibility=flags["Plausibility"], carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, strong_anchor_rgb=base, source_hair_l=source_l, reference_l=reference_l, carrier_stats_mask=trusted)
    assert float(selected_aux["ab_gate_strength"].max()) == 0.0


def main() -> None:
    run_checks()
    print("V2.47.1 appearance correctness tests: PASS (12 checks)")


if __name__ == "__main__":
    main()
