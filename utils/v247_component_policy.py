"""Overall + 2-of-3 chroma-group policy and selected-component strategy."""

from __future__ import annotations


def _decision(summary: dict[str, object], key: str) -> str:
    return str(summary.get(key, {}).get("decision", "NOT_HELPFUL"))


def build_component_policy(overall: dict[str, object], groups: dict[str, dict[str, object]]) -> dict[str, object]:
    policy: dict[str, object] = {}
    for component, summary_key in (("L", "l_module"), ("AB", "ab_module"), ("Shading", "shading_module"), ("Plausibility", "plausibility_module")):
        overall_decision = _decision(overall, summary_key)
        group_decisions = {name: _decision(summary, summary_key) for name, summary in groups.items()}
        useful_count = sum(value.endswith("USEFUL") for value in group_decisions.values())
        overall_useful = overall_decision.endswith("USEFUL")
        global_enabled = overall_useful and useful_count >= 2
        disabled_groups = [name for name, value in group_decisions.items() if value.endswith("HARMFUL")]
        policy[component] = {"overall": overall_decision, "low": group_decisions.get("low_chroma", "NOT_HELPFUL"), "normal": group_decisions.get("normal_chroma", "NOT_HELPFUL"), "high": group_decisions.get("high_chroma", "NOT_HELPFUL"), "global_enabled": global_enabled, "disabled_groups": disabled_groups}
    return policy


def selected_components_for_group(policy: dict[str, object], group: str) -> dict[str, bool]:
    group_key = {"low": "low_chroma", "normal": "normal_chroma", "high": "high_chroma"}.get(group, group)
    return {component: bool(entry.get("global_enabled", False)) and group_key not in entry.get("disabled_groups", []) for component, entry in policy.items()}


__all__ = ["build_component_policy", "selected_components_for_group"]
