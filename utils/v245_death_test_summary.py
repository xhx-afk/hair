"""V2.45.1 unified summary and detailed report writer."""

from __future__ import annotations

import json
from pathlib import Path


def build_summary(*, matte: dict[str, object], appearance: dict[str, object], occluder: dict[str, object]) -> dict[str, object]:
    next_action = []
    if matte.get("mechanism") == "MATTE_BOTTLENECK_CONFIRMED_BUT_PROBE_INSUFFICIENT": next_action.append("develop_better_independent_hair_matte")
    if appearance.get("scene_tint") == "GLOBAL_SCENE_TINT_HARMFUL": next_action.append("disable_global_scene_tint_and_retest_appearance")
    if appearance.get("shading_conditioner") == "SHADING_CONDITIONER_USEFUL": next_action.append("retain_shading_conditioner")
    if occluder.get("z_order_mechanism") == "OCCLUDER_Z_ORDER_MECHANISM_CONFIRMED" and occluder.get("real_parser_coverage") == "REAL_OCCLUDER_COVERAGE_INSUFFICIENT": next_action.append("develop_generic_foreground_occluder_segmentation")
    return {"version": "v2.45.1", "hashes": matte.get("hashes", {}), "matte": {"mechanism": matte.get("mechanism"), "probe_quality": matte.get("probe_quality"), "coverage_effect": matte.get("coverage_effect"), "boundary_effect": matte.get("boundary_effect"), "bleed_effect": matte.get("bleed_effect")}, "appearance": {"photometric_overall": appearance.get("photometric_overall"), "scene_tint": appearance.get("scene_tint"), "shading_conditioner": appearance.get("shading_conditioner"), "carrier_preservation": appearance.get("carrier_preservation"), "reference_fidelity": appearance.get("reference_fidelity")}, "occluder": {"z_order_mechanism": occluder.get("z_order_mechanism"), "real_parser_coverage": occluder.get("real_parser_coverage"), "real_protection": occluder.get("real_protection"), "headphone_supported": False, "earphone_supported": False, "generic_accessory_supported": False}, "next_action": next_action}


def write_summary(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "death_test_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    matte, appearance, occluder = payload["matte"], payload["appearance"], payload["occluder"]
    lines = ["# V2.45.1 Death Test Report", "", "## Matte", f"- Coverage effect: `{matte.get('coverage_effect')}`", f"- Boundary effect: `{matte.get('boundary_effect')}`", f"- Skin bleed effect: `{matte.get('bleed_effect')}`", f"- Flyaway validity: `{matte.get('probe_quality')}`", f"- Decision: `{matte.get('mechanism')}`", "", "## Appearance", f"- A0/A1/A2/A3 comparison: `{appearance.get('photometric_overall')}`", f"- Reference fidelity: `{appearance.get('reference_fidelity')}`", f"- Scene tint conclusion: `{appearance.get('scene_tint')}`", f"- Shading conclusion: `{appearance.get('shading_conditioner')}`", f"- Carrier preservation: `{appearance.get('carrier_preservation')}`", "", "## Occluder", f"- Synthetic Z-order: `{occluder.get('z_order_mechanism')}`", f"- Real parser coverage: `{occluder.get('real_parser_coverage')}`", f"- Real protection: `{occluder.get('real_protection')}`", "- Unsupported classes: headphone=false, earphone=false, generic_accessory=false", "", "## Recommended Next Version"]
    lines.extend(f"- {action}" for action in payload.get("next_action", []))
    if not payload.get("next_action"): lines.append("- No automatic action; inspect the per-sample diagnostics.")
    (root / "DEATH_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["build_summary", "write_summary"]
