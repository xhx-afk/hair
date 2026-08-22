"""Run the deterministic V2.46 Appearance-only death test."""

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
    p.add_argument("--manifest", type=Path, default=Path("res/v245_death_tests/manifest.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--output-root", type=Path, default=Path("res/v246_appearance_test"))
    p.add_argument("--reuse-v245-cache", action="store_true")
    return p.parse_args()


def _value(value: torch.Tensor) -> object:
    flat = value.detach().cpu().reshape(-1)
    return float(flat[0]) if flat.numel() == 1 else [float(x) for x in flat]


def _save_preview(path: Path, tensors: list[torch.Tensor]) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for tensor in tensors:
        tensor = tensor[:1] if tensor.dim() == 4 else tensor.unsqueeze(0)
        if tensor.size(1) == 1: tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.size(1) == 2: tensor = torch.cat((tensor, torch.zeros_like(tensor[:, :1])), dim=1)
        elif tensor.size(1) > 3: tensor = tensor[:, :3]
        tiles.append(tensor.float().clamp(0, 1).cpu())
    save_image(torch.cat(tiles, dim=3)[0], path)


def _save_map(path: Path, tensor: torch.Tensor) -> None:
    value = tensor[:1].detach().float()
    if value.size(1) > 1: value = value.norm(dim=1, keepdim=True)
    low, high = value.flatten(1).amin(1).view(-1, 1, 1, 1), value.flatten(1).amax(1).view(-1, 1, 1, 1)
    _save_preview(path, [((value - low) / (high - low).clamp_min(1e-6)).repeat(1, 3, 1, 1)])


def _load_rows(path: Path, limit: int) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("samples", []))[:max(1, int(limit))]
    if not rows: raise RuntimeError(f"Manifest has no samples: {path}")
    return rows


def _ensure_cache(args: argparse.Namespace, rows: list[dict[str, object]]) -> Path | None:
    cache = args.output_root / "cache"
    if args.reuse_v245_cache:
        candidates = [cache, Path("res/v2451_death_tests/cache"), Path("res/v245_death_tests/cache")]
        source = next((item for item in candidates if item.exists() and any(item.glob("*.pt"))), None)
        if source is None:
            return None
        cache.mkdir(parents=True, exist_ok=True)
        if source != cache:
            for item in source.glob("*.pt"): shutil.copy2(item, cache / item.name)
        return cache
    raise RuntimeError("V2.46 runner requires --reuse-v245-cache so the frozen V2.45 carrier/cache is reused")


def main() -> None:
    args = parse_args(); rows = _load_rows(args.manifest, args.limit); cache_dir = _ensure_cache(args, rows)
    if cache_dir is None:
        # Rebuild the immutable V2.45 carrier cache from the supplied manifest.
        # The V2.45 collector consumes these rows directly; it does not split,
        # randomize, or reorder the sample set.
        from scripts.run_v2451_death_tests import _collect_cache, _setup_runtime
        rebuild_args = argparse.Namespace(profile="small", seed=3407, device=args.device, output_root=args.output_root, reuse_v245_cache=False)
        bt, trainer, post_process, triplets, device = _setup_runtime(rebuild_args, rows)
        _collect_cache(rebuild_args, bt, trainer, post_process, triplets, device, rows)
        cache_dir = args.output_root / "cache"
    from models.appearance_probe_v246 import AppearanceProbeV246
    from models.v245_death_test_common import load_cache
    from utils.v246_appearance_metrics import appearance_metric_tensors, classify_appearance
    probe, records = AppearanceProbeV246(), []
    for index, row in enumerate(rows):
        data = load_cache(cache_dir, row["sample_id"])
        trusted = probe.trusted_core(data["target_hair_mask"], data["anchor_hair_evidence"])
        reference_l_center = data.get("reference_l", data["carrier_lab"][:, :1].flatten(1).median(dim=1).values.view(-1, 1, 1, 1))
        outputs, aux = probe(carrier_lab=data["carrier_lab"], base_rgb=data["base_rgb"], strong_anchor_rgb=data["strong_anchor_rgb"], carrier_rgb=data["carrier_rgb"], target_hair_mask=data["target_hair_mask"], source_hair_mask=data["source_hair_mask"], reference_l_center=reference_l_center, reference_ab_low=data["reference_ab_low"], source_hair_l=data.get("source_hair_l"), reference_l=data.get("reference_l"), scene_context_valid_fraction=data.get("scene_illumination_reliability"), trusted_alpha=trusted, return_aux=True)
        record = {"sample_id": row["sample_id"]}; previews = []
        for name in ("b0_carrier_rgb", "b1_full_rgb", "b2_no_l_rgb", "b3_no_shading_rgb"):
            variant = name.split("_")[0]; rgb = outputs[name]; metrics = appearance_metric_tensors(carrier_rgb=data["carrier_rgb"], output_rgb=rgb, trusted_core=trusted, reference_rgb=data["color_reference_rgb"], reference_hair_mask=data["reference_hair_mask"], aux=aux.get(variant, {})); record.update({f"{variant}_{key}": _value(value) for key, value in metrics.items()}); previews.append((variant, outputs[name.replace("_rgb", "_preview")]))
        visual_dir = args.output_root / "comparisons" / "visual"; appearance_dir = args.output_root / "comparisons" / "appearance"; frequency_dir = args.output_root / "comparisons" / "frequency"; debug_dir = args.output_root / "debug" / f"{index:03d}"; visual_dir.mkdir(parents=True, exist_ok=True); appearance_dir.mkdir(parents=True, exist_ok=True); frequency_dir.mkdir(parents=True, exist_ok=True); debug_dir.mkdir(parents=True, exist_ok=True)
        if index < 8:
            _save_preview(visual_dir / f"sample_{index:03d}.png", [data["base_rgb"], data["base_rgb"], data["color_reference_rgb"], data["base_rgb"], data["strong_anchor_rgb"], outputs["b1_full_rgb"], data["v244_alpha"], outputs["b1_full_preview"]])
            _save_preview(appearance_dir / f"sample_{index:03d}.png", [data["color_reference_rgb"], data["carrier_rgb"], outputs["b1_full_preview"], outputs["b2_no_l_preview"], outputs["b3_no_shading_preview"], aux["b1"]["delta_l_low_bandlimited"], aux["b1"]["final_chroma"], trusted])
            _save_preview(frequency_dir / f"sample_{index:03d}.png", [aux["b1"]["carrier_l"], aux["b1"]["final_l"], aux["b1"]["delta_l_low_bandlimited"], aux["b1"]["final_l_detail"], aux["b1"]["carrier_ab_detail"], aux["b1"]["final_ab"][:, 1:2], (aux["b1"]["final_ab"] - aux["b1"]["carrier_ab"]).abs().mean(1, keepdim=True), trusted])
        for key in ("carrier_l", "carrier_l_low", "desired_l_low", "delta_l_low_raw", "delta_l_low_bandlimited", "final_l", "carrier_ab_low", "carrier_ab_detail", "delta_ab_low", "provisional_ab", "shadow_chroma_scale", "highlight_chroma_scale", "plausibility_scale", "final_chroma", "pre_gamut_scale", "final_gamut_scale", "total_gamut_scale"):
            if key in aux["b1"]: _save_map(debug_dir / f"{key}.png", aux["b1"][key])
        records.append(record)
    summary = classify_appearance(records); metrics_dir = args.output_root / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True); (args.output_root / "per_sample.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in records) + "\n", encoding="utf-8"); (metrics_dir / "appearance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); (metrics_dir / "v246_summary.json").write_text(json.dumps({"version": "v2.46", "scene_tint_enabled": False, **summary}, indent=2), encoding="utf-8")
    categories = {"low_chroma": [], "normal_chroma": [], "high_chroma": []}
    for row in records:
        chroma = float(row.get("b1_reference_hair_chroma", 0.0)); category = "low_chroma" if chroma < 12 else "normal_chroma" if chroma < 25 else "high_chroma"; categories[category].append(row)
    for category, group in categories.items(): (metrics_dir / f"appearance_{category}.json").write_text(json.dumps(classify_appearance(group), indent=2), encoding="utf-8")
    acceptance = {"version": "v2.46", "appearance_decision": summary.get("decision"), "scene_tint_enabled": False, "carrier_preservation": summary.get("carrier_preservation"), "l_residual_bandlimit": summary.get("l_residual_bandlimit"), "reference_fidelity": summary.get("reference_fidelity")}; (args.output_root / "v246_acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8"); print(json.dumps(acceptance, indent=2))


if __name__ == "__main__": main()
