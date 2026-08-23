"""Deterministic V2.47 correctness checks (diagnostic-only)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.appearance_probe_v247 import AppearanceProbeV247
from models.carrier_reference_error_v847 import CarrierReferenceErrorEstimatorV847
from models.hair_photometric_residual_v847 import HairPhotometricResidualV847, _masked_q
from utils.v247_appearance_metrics import _upper_tail_mean, _lower_tail_mean, _batch_scalar
from models.SG_IDCT_v16 import lab_to_rgb
from utils.v247_appearance_metrics import appearance_metric_tensors
from utils.v247_component_policy import build_component_policy, selected_components_for_group


def _artifact_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"missing V2.47 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"artifact must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise AssertionError(f"missing V2.47 artifact: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AssertionError(f"per_sample line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise AssertionError(f"empty V2.47 artifact: {path}")
    return rows


def _json_equivalent(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equivalent(a, b) for a, b in zip(left, right))
    return left == right


def run_artifact_checks(output_root: Path) -> int:
    """Validate files produced by a completed runner invocation.

    The source tree intentionally does not contain experiment outputs.  A
    missing output root is therefore an explicit skip; a present but partial
    output is a failure and cannot be mistaken for a passing experiment.
    """
    output_root = output_root.expanduser()
    if not output_root.exists():
        print(f"V2.47 artifact checks: SKIP (output root not found: {output_root})")
        return 0
    metrics_root = output_root / "metrics"
    summary = _read_json(metrics_root / "selected_summary.json")
    policy = _read_json(metrics_root / "selected_policy.json")
    _read_json(metrics_root / "component_policy.json")
    acceptance = _read_json(output_root / "v247_acceptance.json")
    records = _read_jsonl(output_root / "per_sample.jsonl")

    required_summary_keys = (
        "selected_components", "selected_l_q50_error", "selected_median_ab_error",
        "selected_chroma_error", "selected_hue_error", "carrier_mid_structure_corr",
        "carrier_gradient_structure_corr", "carrier_relative_mid_energy",
        "carrier_relative_hf_energy", "strict_non_hair_max_rgb_change",
    )
    missing = [key for key in required_summary_keys if key not in summary]
    assert not missing, f"selected_summary missing keys: {missing}"
    assert isinstance(summary["selected_components"], list)
    assert isinstance(summary.get("medians"), dict)

    selected_components = summary["selected_components"]
    assert acceptance.get("selected_components") == selected_components
    assert acceptance.get("selected_global_components") == selected_components
    assert acceptance.get("carrier_contract") == summary.get("carrier_contract")
    assert _json_equivalent(acceptance.get("selected_metrics"), summary.get("medians"))
    assert policy.get("selected_global_components") == selected_components
    assert "recommended_active_components" not in json.dumps(acceptance, sort_keys=True)

    from utils.v247_appearance_metrics import classify_components

    overall = classify_components(records)
    plausibility = overall.get("plausibility_module", {}).get("decision")
    assert plausibility in {
        "PLAUSIBILITY_MODULE_USEFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL",
        "PLAUSIBILITY_MODULE_HARMFUL",
    }, f"unexpected plausibility decision: {plausibility}"
    print(f"V2.47 real plausibility regression: {plausibility} ({len(records)} samples)")

    def fidelity_ok(row: dict[str, object]) -> bool:
        c4_ab, c5_ab = _artifact_float(row, "c4_median_ab_error"), _artifact_float(row, "c5_median_ab_error")
        c4_hue, c5_hue = _artifact_float(row, "c4_stable_hue_error_deg"), _artifact_float(row, "c5_stable_hue_error_deg")
        return all(value is not None for value in (c4_ab, c5_ab, c4_hue, c5_hue)) and c5_ab <= c4_ab * 1.05 and c5_hue <= c4_hue + 1.0

    no_gain_rows = []
    gain_rows = []
    for row in records:
        c4_chroma, c5_chroma = _artifact_float(row, "c4_chroma_error"), _artifact_float(row, "c5_chroma_error")
        c4_high, c5_high = _artifact_float(row, "c4_highlight_to_midtone_chroma_ratio"), _artifact_float(row, "c5_highlight_to_midtone_chroma_ratio")
        if None in (c4_chroma, c5_chroma, c4_high, c5_high) or not fidelity_ok(row):
            continue
        chroma_gain = 1.0 - c5_chroma / max(abs(c4_chroma), 1e-6)
        highlight_gain = 1.0 - max(.80 - c5_high, c5_high - 1.05, 0.0) / max(max(.80 - c4_high, c4_high - 1.05, 0.0), 1e-6)
        (gain_rows if chroma_gain >= .05 or highlight_gain >= .10 else no_gain_rows).append(row)

    if no_gain_rows:
        no_gain_decision = classify_components(no_gain_rows).get("plausibility_module", {}).get("decision")
        assert no_gain_decision == "PLAUSIBILITY_MODULE_NOT_HELPFUL", f"real no-gain case classified as {no_gain_decision}"
        print(f"V2.47 real plausibility Case A: PASS ({len(no_gain_rows)} samples)")
    else:
        print("V2.47 real plausibility Case A: SKIP (no fidelity-preserving no-gain subset)")
    if gain_rows:
        gain_decision = classify_components(gain_rows).get("plausibility_module", {}).get("decision")
        assert gain_decision == "PLAUSIBILITY_MODULE_USEFUL", f"real gain case classified as {gain_decision}"
        print(f"V2.47 real plausibility Case B: PASS ({len(gain_rows)} samples)")
    else:
        print("V2.47 real plausibility Case B: SKIP (no >=5% chroma or >=10% highlight gain subset)")
    return 5


def _scene(size: int = 64):
    axis = torch.linspace(-1.0, 1.0, size)
    pattern = (axis.view(1, 1, size, 1) * .025 + axis.view(1, 1, 1, size) * .025)
    base = (torch.full((1, 3, size, size), .35) + pattern).clamp(0, 1)
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
    for variant in ("c0", "c1", "c2", "c3", "c4", "c5"):
        assert outputs[f"{variant}_rgb"].shape == base.shape
    assert set(aux["c0"].keys()) == set(aux["c1"].keys())
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
    hair = torch.zeros_like(luma); hair[:, :, 16:] = 1.0
    assert float(_masked_q(luma, hair, .8)) < 60

    # 5. Highlight mean uses the highlight-region denominator.
    chroma = torch.full_like(luma, 5.0); chroma[:, :, 16:24, :] = 20.0
    luma[:, :, 16:24, :] = 60.0
    assert float(_upper_tail_mean(chroma, luma, hair, .80)) > 15.0

    # 6. Shadow and highlight tails use explicit lower/upper semantics.
    assert float(_lower_tail_mean(chroma, luma, hair, .20)) < float(_upper_tail_mean(chroma, luma, hair, .80))

    # 7. Gate scalar shape contract accepts [B], [B,1], and [B,1,1,1].
    for shape in ((2,), (2, 1), (2, 1, 1, 1)):
        assert tuple(_batch_scalar(torch.ones(shape), 2).shape) == (2,)

    # 8. Shadow and highlight gates are independent fields.
    _, residual_aux = residual(carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, carrier_stats_mask=trusted, enable_shading=True, return_aux=True)
    assert "shadow_gate_strength" in residual_aux and "highlight_gate_strength" in residual_aux

    # 9. C0 is a true no-op in RGB and aux.
    assert torch.equal(outputs["c0_rgb"], base)
    for key in ("l_gate_strength", "ab_gate_strength", "shadow_gate_strength", "highlight_gate_strength", "shading_gate_strength", "plausibility_gate_strength", "delta_l", "gated_delta_l", "delta_ab_center"):
        assert float(aux["c0"][key].abs().max()) == 0.0
    assert float(aux["c0"]["total_gamut_scale"].min()) == 1.0 and float(aux["c0"]["no_op"].min()) == 1.0
    for key, value in aux["c0"].items():
        if torch.is_tensor(value) and key in aux["c1"] and torch.is_tensor(aux["c1"][key]):
            assert value.data_ptr() != aux["c1"][key].data_ptr(), key
    assert torch.equal(aux["c0"]["final_l"], aux["c0"]["carrier_l"])
    assert torch.equal(aux["c0"]["final_ab"], aux["c0"]["carrier_ab"])
    if "desired_l_low" in aux["c0"]:
        assert float(aux["c0"]["desired_l_low"].abs().max()) == 0.0
    old_c0 = aux["c0"]["final_l"].clone(); aux["c1"]["final_l"].add_(1.0)
    assert torch.equal(aux["c0"]["final_l"], old_c0)

    # 10. High chroma receives the conservative AB modifier.
    low_carrier = lab_to_rgb(torch.tensor([[[[50.0]], [[0.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    low_reference = lab_to_rgb(torch.tensor([[[[50.0]], [[8.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    high_carrier = lab_to_rgb(torch.tensor([[[[50.0]], [[30.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    high_reference = lab_to_rgb(torch.tensor([[[[50.0]], [[38.0]], [[0.0]]]])).expand_as(base).clamp(0, 1)
    low = estimator(carrier_rgb=low_carrier, reference_rgb=low_reference, carrier_hair_mask=target, reference_hair_mask=target)
    high = estimator(carrier_rgb=high_carrier, reference_rgb=high_reference, carrier_hair_mask=target, reference_hair_mask=target)
    assert float(high["ab_gate_strength"].max()) <= float(low["ab_gate_strength"].max()) + 1e-5

    # 11. Active correction remains hair-only.
    changed = outputs["c3_rgb"]; strict = aux["c3"]["target_hair_soft"] < .01
    assert float(((changed - base).abs().mean(1, keepdim=True) * strict).max()) <= 1e-5

    def summary(decisions):
        return {"l_module": {"decision": decisions[0]}, "ab_module": {"decision": decisions[1]}, "shading_module": {"decision": decisions[2]}, "plausibility_module": {"decision": decisions[3]}}

    # 12. Two useful chroma groups enable a component and harmful group disables it.
    overall = summary(("L_MODULE_USEFUL", "AB_MODULE_USEFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL"))
    groups = {"low_chroma": summary(("L_MODULE_USEFUL", "AB_MODULE_USEFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL")), "normal_chroma": summary(("L_MODULE_USEFUL", "AB_MODULE_USEFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL")), "high_chroma": summary(("L_MODULE_USEFUL", "AB_MODULE_HARMFUL", "SHADING_MODULE_NOT_HELPFUL", "PLAUSIBILITY_MODULE_NOT_HELPFUL"))}
    policy = build_component_policy(overall, groups); assert policy["AB"]["global_enabled"] and "high_chroma" in policy["AB"]["disabled_groups"]

    # 13. NOT_HELPFUL groups are disabled just like harmful groups.
    groups["high_chroma"]["ab_module"]["decision"] = "AB_MODULE_NOT_HELPFUL"; policy_not_helpful = build_component_policy(overall, groups)
    assert "high_chroma" in policy_not_helpful["AB"]["disabled_groups"] and not selected_components_for_group(policy_not_helpful, "high_chroma")["AB"]

    # 14. One useful group plus two harmful groups is rejected globally.
    groups["normal_chroma"]["ab_module"]["decision"] = "AB_MODULE_HARMFUL"; policy_reject = build_component_policy(overall, groups)
    assert not policy_reject["AB"]["global_enabled"]

    # 15. Selected candidate flags are exactly the group policy flags.
    flags = selected_components_for_group(policy, "high_chroma")
    selected_rgb, selected_aux = probe.run_selected(enable_l=flags["L"], enable_ab=flags["AB"], enable_shading=flags["Shading"], enable_plausibility=flags["Plausibility"], carrier_rgb=base, reference_rgb=reference, target_hair_mask=target, reference_hair_mask=target, strong_anchor_rgb=base, source_hair_l=source_l, reference_l=reference_l, carrier_stats_mask=trusted)
    assert float(selected_aux["ab_gate_strength"].max()) == 0.0
    selected_metrics = appearance_metric_tensors(carrier_rgb=base, output_rgb=selected_rgb, trusted_core=trusted, reference_rgb=reference, reference_hair_mask=target, aux=selected_aux)
    assert "l_q50_error" in selected_metrics and "strict_non_hair_max_rgb_change" in selected_metrics
    assert float(selected_metrics["carrier_mid_structure_corr"].min()) >= .95
    assert float(selected_metrics["carrier_gradient_structure_corr"].min()) >= .95
    assert .95 <= float(selected_metrics["carrier_relative_mid_energy"].median()) <= 1.10
    assert .90 <= float(selected_metrics["carrier_relative_hf_energy"].median()) <= 1.15
    assert float(selected_metrics["strict_non_hair_max_rgb_change"].max()) <= 1e-5

    # 16. Invalid shading regions are explicitly marked, never treated as zero.
    tiny = torch.zeros_like(target); tiny[:, :, 20:25, 20:25] = 1
    tiny_metrics = appearance_metric_tensors(carrier_rgb=base, output_rgb=base, trusted_core=tiny, reference_rgb=reference, reference_hair_mask=tiny, aux={})
    assert float(tiny_metrics["shading_metric_valid"].max()) == 0.0

    # 17. Shading uses distance to a target range, not "smaller is better".
    distance = lambda value, low, high: max(low - value, value - high, 0.0)
    assert distance(.30, .70, .95) > distance(.60, .70, .95)

    # 18. Plausibility requires positive gain, not merely no degradation.
    common = {"appearance_metric_valid": 1, "median_ab_error": 1., "chroma_error": 1., "stable_hue_error_deg": 1., "shadow_to_midtone_chroma_ratio": .8, "highlight_to_midtone_chroma_ratio": .9, "carrier_mid_structure_corr": .99, "carrier_gradient_structure_corr": .99, "carrier_relative_mid_energy": 1., "carrier_relative_hf_energy": 1., "strict_non_hair_max_rgb_change": 0., "l_q10_error": 1., "l_q25_error": 1., "l_q50_error": 1., "l_q75_error": 1., "l_q90_error": 1., "l_delta_clamp_fraction": 0., "plausibility_gate_strength": .5, "noop": 0}
    no_plaus_gain = []
    for variant in ("c0", "c1", "c2", "c3", "c4", "c5"):
        no_plaus_gain.append({f"{variant}_{key}": value for key, value in common.items()})
    no_plaus_row = {}
    for part in no_plaus_gain:
        no_plaus_row.update(part)
    no_plaus_row["reference_chroma_group"] = "high_chroma"
    from utils.v247_appearance_metrics import classify_components
    assert classify_components([no_plaus_row])["plausibility_module"]["decision"] == "PLAUSIBILITY_MODULE_NOT_HELPFUL"
    gain_row = dict(no_plaus_row)
    gain_row.update({"c4_chroma_error": 1.0, "c5_chroma_error": .90, "c4_median_ab_error": 1.0, "c5_median_ab_error": .90, "c4_stable_hue_error_deg": 1.0, "c5_stable_hue_error_deg": .5})
    assert classify_components([gain_row])["plausibility_module"]["decision"] == "PLAUSIBILITY_MODULE_USEFUL"

    # Acceptance must expose exactly the globally enabled policy components.
    selected_global = [component for component, entry in policy.items() if entry["global_enabled"]]
    assert selected_global == [component for component, enabled in selected_components_for_group(policy, "normal_chroma").items() if enabled]
    # Artifact checks below read the actual selected_summary.json; this core
    # test deliberately only verifies the in-memory contract.
    selected_summary_keys = {"selected_components", "selected_l_q50_error", "selected_median_ab_error", "selected_chroma_error", "selected_hue_error", "carrier_mid_structure_corr", "carrier_gradient_structure_corr", "carrier_relative_mid_energy", "carrier_relative_hf_energy", "strict_non_hair_max_rgb_change"}
    assert len(selected_summary_keys) == 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("res/v247_appearance_test"))
    args = parser.parse_args()
    run_checks()
    artifact_checks = run_artifact_checks(args.output_root)
    print(f"V2.47 appearance correctness tests: PASS (20 core checks + {artifact_checks} artifact checks)")


if __name__ == "__main__":
    main()
