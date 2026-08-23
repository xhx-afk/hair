"""Run the deterministic V2.47 carrier-first component death test."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("res/v245_death_tests/manifest.json")); p.add_argument("--device", default="cuda"); p.add_argument("--limit", type=int, default=64); p.add_argument("--output-root", type=Path, default=Path("res/v247_appearance_test")); p.add_argument("--reuse-v245-cache", action="store_true")
    return p.parse_args()


def _load_rows(path: Path, limit: int) -> list[dict[str, object]]:
    rows = list(json.loads(path.read_text(encoding="utf-8")).get("samples", []))[:max(1, int(limit))]
    if not rows: raise RuntimeError(f"Manifest has no samples: {path}")
    return rows


def _ensure_cache(args: argparse.Namespace, rows: list[dict[str, object]]) -> Path:
    cache = args.output_root / "cache"; sources = [cache, Path("res/v2451_death_tests/cache"), Path("res/v245_death_tests/cache")]
    source = next((item for item in sources if item.exists() and any(item.glob("*.pt"))), None)
    if source is not None:
        cache.mkdir(parents=True, exist_ok=True)
        if source != cache:
            for item in source.glob("*.pt"): shutil.copy2(item, cache / item.name)
        return cache
    from scripts.run_v2451_death_tests import _collect_cache, _setup_runtime
    rebuild = argparse.Namespace(profile="small", seed=3407, device=args.device, output_root=args.output_root, reuse_v245_cache=False)
    bt, trainer, post_process, triplets, device = _setup_runtime(rebuild, rows); _collect_cache(rebuild, bt, trainer, post_process, triplets, device, rows)
    return cache


def _value(value: torch.Tensor) -> object:
    flat = value.detach().cpu().reshape(-1)
    return float(flat[0]) if flat.numel() == 1 else [float(item) for item in flat]


def _save_preview(path: Path, tensors: list[torch.Tensor]) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True); tiles = []
    for tensor in tensors:
        tensor = tensor[:1] if tensor.dim() == 4 else tensor.unsqueeze(0)
        if tensor.size(1) == 1: tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.size(1) == 2: tensor = torch.cat((tensor, torch.zeros_like(tensor[:, :1])), 1)
        elif tensor.size(1) > 3: tensor = tensor[:, :3]
        tiles.append(tensor.float().clamp(0, 1).cpu())
    save_image(torch.cat(tiles, dim=3)[0], path)


def _image(data: dict[str, object], *keys: str) -> torch.Tensor:
    for key in keys:
        value = data.get(key)
        if torch.is_tensor(value):
            return value
    return data["base_rgb"]


def _normalized(value: torch.Tensor, signed: bool = False) -> torch.Tensor:
    value = value.float()
    if signed: return (.5 + value / (2.0 * value.abs().flatten(1).amax(1).view(-1, 1, 1, 1).clamp_min(1e-4))).clamp(0, 1)
    if value.size(1) > 1: value = value.norm(dim=1, keepdim=True)
    lo, hi = value.flatten(1).amin(1).view(-1, 1, 1, 1), value.flatten(1).amax(1).view(-1, 1, 1, 1)
    return ((value - lo) / (hi - lo).clamp_min(1e-4)).clamp(0, 1)


def main() -> None:
    args = parse_args(); rows = _load_rows(args.manifest, args.limit); cache_dir = _ensure_cache(args, rows)
    from models.appearance_probe_v247 import AppearanceProbeV247
    from models.v245_death_test_common import composite, load_cache
    from utils.v247_appearance_metrics import appearance_metric_tensors, classify_components
    from utils.v247_component_policy import build_component_policy, selected_components_for_group
    probe, records = AppearanceProbeV247(), []
    for index, row in enumerate(rows):
        data = load_cache(cache_dir, row["sample_id"]); trusted = probe.trusted_core(data["target_hair_mask"], data["anchor_hair_evidence"])
        outputs, aux = probe(carrier_rgb=data["carrier_rgb"], reference_rgb=data["color_reference_rgb"], target_hair_mask=data["target_hair_mask"], reference_hair_mask=data["reference_hair_mask"], strong_anchor_rgb=data["strong_anchor_rgb"], source_hair_l=data.get("source_hair_l"), reference_l=data.get("reference_l"), trusted_alpha=trusted, return_aux=True)
        record = {"sample_id": row["sample_id"]}; previews = []
        for name in ("c0_rgb", "c1_rgb", "c2_rgb", "c3_rgb", "c4_rgb", "c5_rgb"):
            variant = name.split("_")[0]; metrics = appearance_metric_tensors(carrier_rgb=data["carrier_rgb"], output_rgb=outputs[name], trusted_core=trusted, reference_rgb=data["color_reference_rgb"], reference_hair_mask=data["reference_hair_mask"], aux=aux[variant]); record.update({f"{variant}_{key}": _value(value) for key, value in metrics.items()}); previews.append(outputs[f"{variant}_preview"])
        visual_dir = args.output_root / "comparisons" / "visual"; probe_dir = args.output_root / "comparisons" / "appearance_probe"; freq_dir = args.output_root / "comparisons" / "frequency"; debug_dir = args.output_root / "debug" / f"{index:03d}"; visual_dir.mkdir(parents=True, exist_ok=True); probe_dir.mkdir(parents=True, exist_ok=True); freq_dir.mkdir(parents=True, exist_ok=True); debug_dir.mkdir(parents=True, exist_ok=True)
        if index < 8:
            _save_preview(visual_dir / f"sample_{index:03d}.png", [_image(data, "source_rgb", "base_rgb"), _image(data, "shape_reference_rgb", "strong_anchor_rgb", "base_rgb"), data["color_reference_rgb"], data["base_rgb"], data["strong_anchor_rgb"], outputs["c5_rgb"], data["v244_alpha"], composite(data["base_rgb"], outputs["c5_rgb"], data["v244_alpha"])])
            _save_preview(probe_dir / f"sample_{index:03d}.png", [data["color_reference_rgb"], *previews, trusted])
            _save_preview(freq_dir / f"sample_{index:03d}.png", [_normalized(aux["c5"]["carrier_l"] / 100.0), _normalized(aux["c5"]["final_l"] / 100.0), _normalized(aux["c5"]["gated_delta_l"], True), _normalized(aux["c5"]["delta_ab_center"], True), _normalized(aux["c5"]["carrier_ab_detail"], True), _normalized(aux["c5"]["final_ab"], True), trusted])
        for key in ("carrier_l", "carrier_l_low", "desired_l_low", "delta_l_low_raw", "delta_l", "gated_delta_l", "final_l", "carrier_ab_low", "carrier_ab_detail", "delta_ab_center", "final_ab_low", "provisional_ab", "shadow_chroma_scale", "highlight_chroma_scale", "target_hair_soft", "total_gamut_scale"):
            if key in aux["c5"]: _save_preview(debug_dir / f"{key}.png", [_normalized(aux["c5"][key], signed=("delta" in key or "detail" in key))])
        record["stats_mask_source"] = aux["c5"].get("stats_mask_source", "unknown"); record["stats_mask_pixel_count"] = float(aux["c5"].get("stats_mask_pixel_count", aux["c5"].get("stats_mask", trusted).flatten(1).gt(.5).sum(1)).flatten()[0]); record["gate_metric_valid"] = float(aux["c5"].get("gate_metric_valid", torch.zeros(1, device=trusted.device)).flatten()[0]); record["reference_chroma_group"] = "low_chroma" if float(record.get("c0_reference_chroma_median", 0.0)) < 12 else "normal_chroma" if float(record.get("c0_reference_chroma_median", 0.0)) < 25 else "high_chroma"
        records.append(record)
    summary = classify_components(records); metrics_dir = args.output_root / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True); (metrics_dir / "component_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    groups = {"low_chroma": [], "normal_chroma": [], "high_chroma": []}
    for row in records:
        chroma = float(row.get("c0_reference_chroma_median", 0.0)); groups["low_chroma" if chroma < 12 else "normal_chroma" if chroma < 25 else "high_chroma"].append(row)
    group_summaries = {name: classify_components(group) for name, group in groups.items()}
    for name, group_summary in group_summaries.items(): (metrics_dir / f"{name}.json").write_text(json.dumps(group_summary, indent=2), encoding="utf-8")
    policy = build_component_policy(summary, group_summaries); (metrics_dir / "component_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    selected_manifest = []
    selected_metric_keys = ("l_q10_error", "l_q25_error", "l_q50_error", "l_q75_error", "l_q90_error", "median_ab_error", "chroma_error", "stable_hue_error_deg", "carrier_mid_structure_corr", "carrier_gradient_structure_corr", "carrier_relative_mid_energy", "carrier_relative_hf_energy", "shadow_to_midtone_chroma_ratio", "highlight_to_midtone_chroma_ratio", "strict_non_hair_max_rgb_change", "trusted_core_gamut_heavy_fraction", "target_hair_gamut_heavy_fraction", "noop")
    for index, row in enumerate(rows):
        data = load_cache(cache_dir, row["sample_id"]); trusted = probe.trusted_core(data["target_hair_mask"], data["anchor_hair_evidence"]); group = records[index]["reference_chroma_group"]; flags = selected_components_for_group(policy, group); selected, selected_aux = probe.run_selected(enable_l=flags["L"], enable_ab=flags["AB"], enable_shading=flags["Shading"], enable_plausibility=flags["Plausibility"], carrier_rgb=data["carrier_rgb"], reference_rgb=data["color_reference_rgb"], target_hair_mask=data["target_hair_mask"], reference_hair_mask=data["reference_hair_mask"], strong_anchor_rgb=data["strong_anchor_rgb"], source_hair_l=data.get("source_hair_l"), reference_l=data.get("reference_l"), carrier_stats_mask=trusted)
        selected_metrics = appearance_metric_tensors(carrier_rgb=data["carrier_rgb"], output_rgb=selected, trusted_core=trusted, reference_rgb=data["color_reference_rgb"], reference_hair_mask=data["reference_hair_mask"], aux=selected_aux)
        records[index]["selected_components_for_sample"] = [name for name, enabled in flags.items() if enabled]; records[index]["selected_candidate_stats_mask_source"] = selected_aux.get("stats_mask_source", "unknown"); records[index]["selected_metric_valid"] = _value(selected_metrics.get("appearance_metric_valid", torch.zeros(1, device=trusted.device)))
        for key in selected_metric_keys:
            if key in selected_metrics: records[index][f"selected_{key}"] = _value(selected_metrics[key])
        selected_manifest.append({"sample_id": row["sample_id"], "group": group, "components": records[index]["selected_components_for_sample"]})
        if index < 8:
            visual_dir = args.output_root / "comparisons" / "visual"; _save_preview(visual_dir / f"sample_{index:03d}.png", [_image(data, "source_rgb", "base_rgb"), _image(data, "shape_reference_rgb", "strong_anchor_rgb", "base_rgb"), data["color_reference_rgb"], data["base_rgb"], data["strong_anchor_rgb"], selected, data["v244_alpha"], composite(data["base_rgb"], selected, data["v244_alpha"])])
    selected_valid = [row for row in records if float(row.get("selected_metric_valid", 0.0)) > .5]
    def selected_median(key):
        values = [float(row[f"selected_{key}"]) for row in selected_valid if isinstance(row.get(f"selected_{key}"), (int, float))]
        return float(torch.tensor(values).median()) if values else float("nan")
    selected_contract = selected_valid and selected_median("carrier_mid_structure_corr") >= .95 and selected_median("carrier_gradient_structure_corr") >= .95 and .95 <= selected_median("carrier_relative_mid_energy") <= 1.10 and .90 <= selected_median("carrier_relative_hf_energy") <= 1.15 and selected_median("strict_non_hair_max_rgb_change") <= 1e-5
    selected_components_union = sorted({component for row in records for component in row.get("selected_components_for_sample", [])})
    selected_medians = {key: selected_median(key) for key in selected_metric_keys}
    selected_summary = {"count": len(selected_valid), "valid_fraction": len(selected_valid) / max(len(records), 1), "selected_components": selected_components_union, "selected_l_q50_error": selected_medians.get("l_q50_error", float("nan")), "selected_median_ab_error": selected_medians.get("median_ab_error", float("nan")), "selected_chroma_error": selected_medians.get("chroma_error", float("nan")), "selected_hue_error": selected_medians.get("stable_hue_error_deg", float("nan")), "selected_carrier_mid_structure_corr": selected_medians.get("carrier_mid_structure_corr", float("nan")), "selected_gradient_structure_corr": selected_medians.get("carrier_gradient_structure_corr", float("nan")), "selected_strict_non_hair_max_rgb_change": selected_medians.get("strict_non_hair_max_rgb_change", float("nan")), "selected_gamut_fraction": selected_medians.get("target_hair_gamut_heavy_fraction", float("nan")), "medians": selected_medians, "carrier_contract": "PASS" if selected_contract else "FAIL"}
    (metrics_dir / "selected_summary.json").write_text(json.dumps(selected_summary, indent=2, allow_nan=True), encoding="utf-8")
    selected_global_components = [component for component, entry in policy.items() if entry["global_enabled"]]
    if not selected_contract: final_decision = "V247_SELECTED_CARRIER_CONTRACT_FAIL"
    elif not selected_global_components: final_decision = "V247_CARRIER_ONLY"
    elif selected_global_components == ["AB"]: final_decision = "V247_ERROR_AWARE_AB_ONLY"
    elif selected_global_components == ["L"]: final_decision = "V247_ERROR_AWARE_L_ONLY"
    elif set(("L", "AB", "Shading")).issubset(selected_global_components): final_decision = "V247_ERROR_AWARE_L_AB_SHADING"
    else: final_decision = "V247_ERROR_AWARE_L_AB"
    (args.output_root / "selected_policy.json").write_text(json.dumps({"policy": policy, "samples": selected_manifest}, indent=2), encoding="utf-8"); (args.output_root / "per_sample.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=True, allow_nan=True) for row in records) + "\n", encoding="utf-8")
    c0_selected_l = float(torch.tensor([float(row.get("c0_l_q50_error", float("inf"))) for row in selected_valid]).median()) if selected_valid else float("nan")
    reference_gain = bool(selected_valid) and selected_median("l_q50_error") <= c0_selected_l * .95
    acceptance = {"version": "v2.47", "diagnostic_only": True, "component_attribution": {"L": summary.get("l_module", {}).get("decision"), "AB": summary.get("ab_module", {}).get("decision"), "Shading": summary.get("shading_module", {}).get("decision"), "Plausibility": summary.get("plausibility_module", {}).get("decision")}, "component_policy": policy, "selected_global_components": selected_global_components, "selected_candidate_metrics": selected_summary, "selected_candidate": {"carrier_contract": selected_summary["carrier_contract"], "reference_fidelity": "PASS" if reference_gain else ("NO_GAIN" if selected_valid else "FAIL"), "hair_only_contract": "PASS" if selected_median("strict_non_hair_max_rgb_change") <= 1e-5 else "FAIL", "gamut": "PASS" if selected_valid else "FAIL"}, "overall_decision": final_decision, "selected_summary": selected_summary}
    (args.output_root / "v247_acceptance.json").write_text(json.dumps(acceptance, indent=2, allow_nan=True), encoding="utf-8"); print(json.dumps(acceptance, indent=2, allow_nan=True))


if __name__ == "__main__": main()
