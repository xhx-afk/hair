"""Run the isolated V2.45.1 Matte, Appearance, and Occluder death tests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("matte", "appearance", "occluder", "all"), default="all")
    p.add_argument("--manifest", type=Path, default=Path("res/v245_death_tests/manifest.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile", default="small")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--output-root", type=Path, default=Path("res/v2451_death_tests"))
    p.add_argument("--reuse-v245-cache", "--reuse-v244-cache", dest="reuse_v245_cache", action="store_true")
    return p.parse_args()


def _save_preview(path: Path, tensors: list[torch.Tensor]) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for value in tensors:
        value = value[:1] if value.dim() == 4 else value.unsqueeze(0)
        if value.size(1) == 1: value = value.repeat(1, 3, 1, 1)
        tiles.append(value.float().clamp(0, 1).cpu())
    save_image(torch.cat(tiles, dim=3)[0], path)


def _mask(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp(0, 1)
    return value.repeat(1, 3, 1, 1)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n", encoding="utf-8")


def _metric_json_value(value: torch.Tensor) -> object:
    value = value.detach().cpu().reshape(-1)
    if value.numel() == 1:
        return float(value[0])
    return [float(item) for item in value]


def _manifest_rows(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.exists(): raise FileNotFoundError(f"Required frozen manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("samples", []))
    if not rows: raise RuntimeError(f"Manifest has no samples: {path}")
    return rows[:max(int(limit), 1)]


def _setup_runtime(args: argparse.Namespace, rows: list[dict[str, object]]):
    os.environ["BLENDING_V8_DATASET_PROFILE"] = args.profile
    os.environ["BLENDING_V244_DIAGNOSTIC_ONLY"] = "1"
    os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from torch.utils.data import DataLoader
    from scripts import blending_train_v8 as bt
    bt.set_seed(args.seed)
    triplets = [tuple([row["source"], row["shape_reference"], row["color_reference"]]) for row in rows]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    helper = bt.MaskPrepHelper(device)
    dataset = bt.BlendingDatasetV8(triplets, bt.ACTIVE_DATASET_DIR, bt.ACTIVE_FACE_ROOT, bt.ACTIVE_COLOR_ROOT, None)
    loader = DataLoader(dataset, batch_size=bt.USER_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    model, _ = bt.Blending_v8.load_v226_projector_for_diagnostic(bt.USER_V227_BASE_CHECKPOINT, device)
    trainer = bt.BlendingTrainerV8(model, None, loader, loader, helper)
    post_process = bt.PostProcessModel().to(device).eval()
    state = torch.load(bt.USER_V227_PP_CHECKPOINT, map_location=device)
    post_process.load_state_dict(state["model_state_dict"])
    return bt, trainer, post_process, triplets, device


def _collect_cache(args, bt, trainer, post_process, triplets, device, rows):
    from models.v841_runtime_inputs import build_v841_runtime_inputs
    from models.v245_death_test_common import save_cache, write_manifest, tensor_sha256
    cache_dir, root = args.output_root / "cache", args.output_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_v245_cache:
        source = Path("res/v245_death_tests/cache")
        source_manifest = args.manifest
        if source.exists() and source_manifest.exists():
            for item in source.glob("*.pt"):
                target = cache_dir / item.name
                if not target.exists(): shutil.copy2(item, target)
            write_manifest(root / "manifest.json", rows)
            return rows
    wanted = {json.dumps(list(t), separators=(",", ":")): row for t, row in zip(triplets, rows)}
    produced = []
    for batch in trainer.val_loader:
        prepared = trainer.prepare_batch(batch)
        if prepared is None: continue
        base, anchor, _, _, _ = trainer._render_v231_baselines(prepared, post_process)
        runtime = build_v841_runtime_inputs(base_rgb=base, strong_anchor_rgb=anchor, color_reference_rgb=((prepared["color_i"] + 1.0) / 2.0).clamp(0, 1), target_illumination_rgb=base, coarse_target_hair_mask=prepared["v224_target_hair_mask"], source_face_mask=prepared["v230_source_face_mask"], source_skin_mask=prepared["v230_source_skin_mask"], source_hair_mask=prepared.get("v230_source_hair_mask", torch.zeros_like(prepared["v230_source_face_mask"])), reference_hair_mask=prepared["reference_hair_mask"])
        final, aux = trainer.v241_transfer(return_aux=True, **runtime)
        for i, sid_value in enumerate(prepared["sample_id"]):
            sid = str(sid_value)
            try: triplet = json.loads(sid)
            except (TypeError, json.JSONDecodeError): triplet = list(triplets[min(i, len(triplets) - 1)])
            key = json.dumps(list(triplet), separators=(",", ":"))
            if key not in wanted: continue
            row = wanted[key]
            payload = {"base_rgb": base[i:i+1], "strong_anchor_rgb": anchor[i:i+1], "carrier_rgb": aux["new_hair_rgb_v242_carrier"][i:i+1], "target_hair_mask": runtime["coarse_target_hair_mask"][i:i+1], "source_hair_mask": runtime["source_hair_mask"][i:i+1], "source_face_mask": runtime["source_face_mask"][i:i+1], "source_skin_mask": runtime["source_skin_mask"][i:i+1], "reference_hair_mask": runtime["reference_hair_mask"][i:i+1], "v244_alpha": aux["target_hair_alpha_final"][i:i+1], "v244_final": final[i:i+1], "anchor_hair_evidence": aux["anchor_hair_evidence"][i:i+1], "strand_structure_confidence": aux["strand_structure_confidence"][i:i+1], "distance_prior": aux["distance_prior"][i:i+1], "parser_labels": prepared["v230_parser_labels"][i:i+1], "carrier_lab": aux["carrier_lab"][i:i+1], "desired_l_low": aux["desired_l_low"][i:i+1], "reference_ab_low": aux["target_ab_low"][i:i+1], "scene_illumination_ab": aux["scene_illumination_ab"][i:i+1], "source_hair_l": aux.get("source_hair_l", torch.zeros_like(aux["target_hair_alpha_final"]))[i:i+1], "reference_l": aux.get("l_q50", torch.zeros_like(aux["target_hair_alpha_final"]))[i:i+1], "color_reference_rgb": runtime["color_reference_rgb"][i:i+1], "scene_illumination_reliability": aux.get("scene_illumination_reliability", torch.ones_like(aux["target_hair_alpha_final"]))[i:i+1], "shadow_chroma_scale": aux["shadow_chroma_scale"][i:i+1], "highlight_chroma_scale": aux["highlight_chroma_scale"][i:i+1]}
            for name in ("total_gamut_scale", "pre_gamut_scale", "final_gamut_scale"):
                if name in aux: payload[name] = aux[name][i:i+1]
            save_cache(cache_dir, key, payload)
            produced.append(row)
    if len(produced) != len(rows): raise RuntimeError(f"Frozen manifest/cache mismatch: expected {len(rows)}, produced {len(produced)}")
    write_manifest(root / "manifest.json", produced)
    return produced


def _run_matte(args, rows):
    from models.matte_probe_v245 import MatteProbeV245
    from models.v245_death_test_common import composite, load_cache
    from utils.v2451_independent_flyaway import independent_flyaway_candidate
    from utils.v245_matte_metrics import aggregate_matte, matte_metric_tensors
    probe, records = MatteProbeV245(), []
    for i, row in enumerate(rows):
        d = load_cache(args.output_root / "cache", row["sample_id"]); candidate, _ = independent_flyaway_candidate(d["strong_anchor_rgb"], d["target_hair_mask"], d["source_skin_mask"])
        m1, aux = probe(coarse_target_hair_mask=d["target_hair_mask"], distance_prior=d["distance_prior"], anchor_hair_evidence=d["anchor_hair_evidence"], strand_structure_confidence=d["strand_structure_confidence"], source_skin_mask=d["source_skin_mask"], strong_anchor_rgb=d["strong_anchor_rgb"], return_aux=True)
        variants = {"m0": d["v244_alpha"], "m1": m1, "m2": d["target_hair_mask"]}; record = {"sample_id": row["sample_id"]}
        for name, alpha in variants.items():
            values = matte_metric_tensors(base_rgb=d["base_rgb"], carrier_rgb=d["carrier_rgb"], alpha=alpha, coarse_target_hair_mask=d["target_hair_mask"], source_face_mask=d["source_face_mask"], source_skin_mask=d["source_skin_mask"], source_hair_mask=d["source_hair_mask"], hair_support=aux["hair_support"], independent_candidate=candidate)
            record.update({f"{name}_{k}": float(v[0]) for k, v in values.items()})
            torch.save({"alpha": alpha.cpu(), "final": composite(d["base_rgb"], d["carrier_rgb"], alpha).cpu()}, args.output_root / "matte" / "debug" / f"{i:03d}_{name}.pt")
        if i < 8:
            _save_preview(args.output_root / "matte" / "control" / f"sample_{i:03d}.png", [_mask(d["target_hair_mask"]), composite(d["base_rgb"], d["carrier_rgb"], d["target_hair_mask"])])
        if i < 8: _save_preview(args.output_root / "matte" / "visual" / f"sample_{i:03d}.png", [d["base_rgb"], d["strong_anchor_rgb"], d["carrier_rgb"], _mask(d["target_hair_mask"]), _mask(d["v244_alpha"]), composite(d["base_rgb"], d["carrier_rgb"], d["v244_alpha"]), _mask(m1), composite(d["base_rgb"], d["carrier_rgb"], m1)])
        records.append(record)
    summary = aggregate_matte(records); _write_jsonl(args.output_root / "matte" / "per_sample.jsonl", records); (args.output_root / "matte" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); return summary


def _run_appearance(args, rows):
    from models.appearance_probe_v245 import AppearanceProbeV245
    from models.v245_death_test_common import composite, load_cache
    from models.SG_IDCT_v16 import rgb_to_lab
    from utils.v245_appearance_metrics import appearance_metric_tensors, classify_appearance
    probe, records, category_rows = AppearanceProbeV245(), [], {"low_chroma": [], "normal_chroma": [], "high_chroma": []}
    for i, row in enumerate(rows):
        d = load_cache(args.output_root / "cache", row["sample_id"]); trusted = probe.trusted_core(d["target_hair_mask"], d["anchor_hair_evidence"])
        outputs, aux = probe(carrier_lab=d["carrier_lab"], desired_l_low=d["desired_l_low"], reference_ab_low=d["reference_ab_low"], scene_illumination_ab=d["scene_illumination_ab"], source_hair_l=d["source_hair_l"], reference_l=d["reference_l"], base_rgb=d["base_rgb"], trusted_alpha=trusted, return_aux=True)
        outputs["a0_carrier_rgb"] = d["carrier_rgb"]; record = {"sample_id": row["sample_id"]}; preview_map = {}
        ref_c = rgb_to_lab(d["color_reference_rgb"])[:, 1:].norm(dim=1, keepdim=True); ref_mask = d["reference_hair_mask"]; chroma = float((ref_c * ref_mask).flatten(1).sum().item() / ref_mask.flatten(1).sum().clamp_min(1).item()); category = "low_chroma" if chroma < 12 else "normal_chroma" if chroma < 25 else "high_chroma"; category_rows[category].append(record)
        previews = []
        for name, rgb in outputs.items():
            if rgb is None: continue
            variant = name.split("_")[0]; a = aux.get(variant, aux.get("a1", {})); metrics = appearance_metric_tensors(carrier_rgb=d["carrier_rgb"], output_rgb=rgb, trusted_core=trusted, reference_rgb=d["color_reference_rgb"], reference_hair_mask=d["reference_hair_mask"], scene_illumination_ab=d["scene_illumination_ab"], scene_reliability=d["scene_illumination_reliability"], shadow_scale=a.get("shadow_chroma_scale"), highlight_scale=a.get("highlight_chroma_scale"), total_gamut_scale=a.get("total_gamut_scale"))
            record.update({f"{variant}_{k}": _metric_json_value(v) for k, v in metrics.items()}); preview_map[variant] = composite(d["base_rgb"], rgb, trusted); preview_dir = args.output_root / "appearance" / "preview" / category; preview_dir.mkdir(parents=True, exist_ok=True); torch.save(preview_map[variant].cpu(), preview_dir / f"{i:03d}_{variant}.pt")
        if i < 8: _save_preview(args.output_root / "appearance" / "visual" / f"sample_{i:03d}.png", [d["base_rgb"], d["color_reference_rgb"], d["strong_anchor_rgb"], preview_map["a0"], preview_map["a1"], preview_map["a2"], preview_map["a3"], _mask(trusted)])
        records.append(record)
    summary = classify_appearance(records); summary["chroma_groups"] = {}
    for category, group in category_rows.items(): summary["chroma_groups"][category] = classify_appearance([r for r in records if r in group])
    _write_jsonl(args.output_root / "appearance" / "per_sample.jsonl", records); (args.output_root / "appearance" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for category, group in category_rows.items(): (args.output_root / "appearance" / f"summary_{category}.json").write_text(json.dumps(summary["chroma_groups"][category], indent=2), encoding="utf-8")
    return summary


def _run_occluder(args, rows):
    from models.occluder_probe_v245 import OccluderProbeV245
    from models.v245_death_test_common import composite, load_cache
    from utils.v245_occluder_metrics import classify_real, classify_synthetic, occluder_metric_tensors
    probe, real_records, synthetic_records = OccluderProbeV245(), [], []
    for i, row in enumerate(rows):
        d = load_cache(args.output_root / "cache", row["sample_id"]); real = probe.soften(probe.real_mask(d["parser_labels"], d["base_rgb"])); synthetic_parts = probe.synthetic_masks(d["base_rgb"]); synthetic = torch.maximum(torch.maximum(synthetic_parts["horizontal_bar"], synthetic_parts["headphone_arc"]), synthetic_parts["earphone_line"]); current_r, protected_r = probe.apply(base_rgb=d["base_rgb"], hair_rgb=d["carrier_rgb"], hair_alpha=d["v244_alpha"], occluder_alpha=real); current_s, protected_s = probe.apply(base_rgb=d["base_rgb"], hair_rgb=d["carrier_rgb"], hair_alpha=d["v244_alpha"], occluder_alpha=probe.soften(synthetic))
        mr = occluder_metric_tensors(base_rgb=d["base_rgb"], current_rgb=current_r, protected_rgb=protected_r, occluder_alpha=real, hair_alpha=d["v244_alpha"], hair_mask=d["target_hair_mask"], real_mask=real, synthetic_mask=torch.zeros_like(real)); ms = occluder_metric_tensors(base_rgb=d["base_rgb"], current_rgb=current_s, protected_rgb=protected_s, occluder_alpha=probe.soften(synthetic), hair_alpha=d["v244_alpha"], hair_mask=d["target_hair_mask"], real_mask=torch.zeros_like(real), synthetic_mask=synthetic)
        rr = {"sample_id": row["sample_id"], **{k: float(v[0]) for k, v in mr.items()}}; sr = {"sample_id": row["sample_id"], **{k: float(v[0]) for k, v in ms.items()}}; real_records.append(rr); synthetic_records.append(sr)
        if i < 8:
            _save_preview(args.output_root / "occluder" / "real" / "visual" / f"sample_{i:03d}.png", [d["base_rgb"], d["carrier_rgb"], _mask(d["v244_alpha"]), _mask(real), current_r, protected_r, (current_r - protected_r).abs(), protected_r * real + d["base_rgb"] * (1 - real)])
            _save_preview(args.output_root / "occluder" / "synthetic" / "visual" / f"sample_{i:03d}.png", [d["base_rgb"], d["carrier_rgb"], _mask(synthetic), current_s, protected_s, (current_s - protected_s).abs()])
    real_summary, synthetic_summary = classify_real(real_records), classify_synthetic(synthetic_records)
    for folder, recs, summary in (("real", real_records, real_summary), ("synthetic", synthetic_records, synthetic_summary)):
        _write_jsonl(args.output_root / "occluder" / folder / "per_sample.jsonl", recs); (args.output_root / "occluder" / folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**synthetic_summary, **real_summary, "z_order_mechanism": synthetic_summary.get("z_order_mechanism"), "real_parser_coverage": real_summary.get("real_parser_coverage"), "real_protection": real_summary.get("real_protection"), "headphone_supported": False, "earphone_supported": False, "generic_accessory_supported": False}


def main() -> None:
    args = parse_args(); rows = _manifest_rows(args.manifest, args.limit); bt, trainer, post_process, triplets, device = _setup_runtime(args, rows); rows = _collect_cache(args, bt, trainer, post_process, triplets, device, rows)
    for folder in ("matte/visual", "matte/debug", "matte/control", "appearance/visual", "appearance/preview", "occluder/real/visual", "occluder/synthetic/visual"): (args.output_root / folder).mkdir(parents=True, exist_ok=True)
    results = {"matte": {"mechanism": "NOT_RUN"}, "appearance": {"photometric_overall": "NOT_RUN"}, "occluder": {"z_order_mechanism": "NOT_RUN"}}
    if args.experiment in ("matte", "all"): results["matte"] = _run_matte(args, rows)
    if args.experiment in ("appearance", "all"): results["appearance"] = _run_appearance(args, rows)
    if args.experiment in ("occluder", "all"): results["occluder"] = _run_occluder(args, rows)
    from models.v245_death_test_common import load_cache, manifest_sha256, tensor_sha256
    first_cache = load_cache(args.output_root / "cache", rows[0]["sample_id"])
    hashes = {"carrier_sha256": tensor_sha256(first_cache["carrier_rgb"]), "alpha_sha256": tensor_sha256(first_cache["v244_alpha"]), "manifest_sha256": manifest_sha256(rows)}
    for result in results.values(): result["hashes"] = hashes
    for path in (args.output_root / "matte" / "summary.json", args.output_root / "appearance" / "summary.json", args.output_root / "occluder" / "real" / "summary.json", args.output_root / "occluder" / "synthetic" / "summary.json"):
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8")); stored["hashes"] = hashes; path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    if args.experiment == "all":
        from utils.v245_death_test_summary import build_summary, write_summary
        write_summary(args.output_root, build_summary(**results))
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__": main()
