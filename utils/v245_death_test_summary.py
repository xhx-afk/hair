"""Unified V2.45 death-test summary and next-action policy."""

from __future__ import annotations

import json
from pathlib import Path


def build_summary(*, matte: dict[str, object], appearance: dict[str, object], occluder: dict[str, object]) -> dict[str, object]:
    decisions = {"matte": matte.get("decision"), "appearance": appearance.get("decision"), "occluder": occluder.get("decision")}
    next_action = []
    if decisions["matte"] not in ("MATTE_BOTTLENECK_CONFIRMED",):
        next_action.append("isolate_and_fix_matte_before_integration")
    if decisions["appearance"] in ("PHOTOMETRIC_MODULE_HARMFUL", "LAB_RECOLOR_LIMIT_CONFIRMED"):
        next_action.append("freeze_matte_and_isolate_appearance")
    if decisions["occluder"] in ("EXPLICIT_OCCLUDER_OWNERSHIP_REQUIRED", "OCCLUDER_MODEL_COVERAGE_INSUFFICIENT"):
        next_action.append("add_explicit_foreground_occluder_ownership")
    if not next_action:
        next_action.append("review_three_death_tests_before_selecting_next_version")
    return {"version": "v2.45", "matte": matte, "appearance": appearance, "occluder": occluder, "next_action": next_action}


def write_summary(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "death_test_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# HairFast V2.45 Death Test Report", "", f"- Matte: `{payload['matte']['decision']}`", f"- Appearance: `{payload['appearance']['decision']}`", f"- Occluder: `{payload['occluder']['decision']}`", "", "## Next Action", ""]
    lines.extend(f"- {item}" for item in payload.get("next_action", []))
    (root / "DEATH_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["build_summary", "write_summary"]
