"""Run the isolated V2.45 Matte, Appearance, and Occluder death tests.

The runner is diagnostic-only. It creates one deterministic manifest and one
frozen V2.44 carrier cache, then each experiment reads only those inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("matte", "appearance", "occluder", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", default="small")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-root", type=Path, default=Path("res/v245_death_tests"))
    parser.add_argument("--reuse-v244-cache", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    return parser.parse_args()


def _json_sample(value: object) -> str:
    return str(value)


def _save_preview(path: Path, tensors: list[torch.Tensor]) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for value in tensors:
        value = value[:1] if value.dim() == 4 else value.unsqueeze(0)
        if value.size(1) == 1:
            value = value.repeat(1, 3, 1, 1)
        tiles.append(value.float().clamp(0, 1).cpu())
    save_image(torch.cat(tiles, dim=3)[0], path)


def _mask_preview(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp(0, 1)
    return value.repeat(1, 3, 1, 1)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _setup_runtime(args: argparse.Namespace):
    os.environ["BLENDING_V8_DATASET_PROFILE"] = args.profile
    os.environ["BLENDING_V244_DIAGNOSTIC_ONLY"] = "1"
    os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from scripts import blending_train_v8 as bt

    bt.set_seed(args.seed)
    triplets = bt.read_triplets(bt.ACTIVE_DATASET_DIR)
    if not triplets:
        raise RuntimeError(f"No triplets found in {bt.ACTIVE_DATASET_DIR / 'dataset.exps'}")
    _, val_exps = train_test_split(triplets, test_size=bt.ACTIVE_VAL_SIZE, random_state=args.seed)
    val_exps = list(val_exps)
    visual_ids = []
    acceptance = Path("res/v244/v244_acceptance.json")
    if acceptance.exists():
        try:
            visual_ids = [str(item) for item in json.loads(acceptance.read_text(encoding="utf-8")).get("visual_sample_ids", [])]
        except Exception:
            visual_ids = []
    if visual_ids:
        preferred = {item for item in visual_ids}
        val_exps = sorted(val_exps, key=lambda row: (json.dumps(list(row), separators=(",", ":")) not in preferred, json.dumps(list(row), separators=(",", ":"))))
    val_exps = val_exps[:max(int(args.limit), 1)]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    helper = bt.MaskPrepHelper(device)
    dataset = bt.BlendingDatasetV8(val_exps, bt.ACTIVE_DATASET_DIR, bt.ACTIVE_FACE_ROOT, bt.ACTIVE_COLOR_ROOT, None)
    loader = DataLoader(dataset, batch_size=bt.USER_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    model, _ = bt.Blending_v8.load_v226_projector_for_diagnostic(bt.USER_V227_BASE_CHECKPOINT, device)
    trainer = bt.BlendingTrainerV8(model, None, loader, loader, helper)
    post_process = bt.PostProcessModel().to(device).eval()
    state = torch.load(bt.USER_V227_PP_CHECKPOINT, map_location=device)
    post_process.load_state_dict(state["model_state_dict"])
    return bt, trainer, post_process, val_exps, device


def _collect_cache(args: argparse.Namespace, bt, trainer, post_process, val_exps, device) -> list[dict[str, object]]:
    from models.v841_runtime_inputs import build_v841_runtime_inputs
    from models.v245_death_test_common import save_cache, write_manifest
    root = args.output_root
    cache_dir = root / "cache"
    manifest_path = root / "manifest.json"
    if args.reuse_v244_cache and manifest_path.exists() and any(cache_dir.glob("*.pt")):
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("samples", [])
    rows, wanted = [], {json.dumps(list(row), ensure_ascii=True, separators=(",", ":")) for row in val_exps}
    for batch in trainer.val_loader:
        prepared = trainer.prepare_batch(batch)
        if prepared is None:
            continue
        base, anchor, _, _, _ = trainer._render_v231_baselines(prepared, post_process)
        runtime = build_v841_runtime_inputs(
            base_rgb=base, strong_anchor_rgb=anchor,
            color_reference_rgb=((prepared["color_i"] + 1.0) / 2.0).clamp(0, 1),
            target_illumination_rgb=base, coarse_target_hair_mask=prepared["v224_target_hair_mask"],
            source_face_mask=prepared["v230_source_face_mask"], source_skin_mask=prepared["v230_source_skin_mask"],
            source_hair_mask=prepared.get("v230_source_hair_mask", torch.zeros_like(prepared["v230_source_face_mask"])),
            reference_hair_mask=prepared["reference_hair_mask"],
        )
        final, aux = trainer.v241_transfer(return_aux=True, **runtime)
        for index, sample_id in enumerate(prepared["sample_id"]):
            sid = _json_sample(sample_id)
            try:
                triplet = json.loads(sid)
            except (TypeError, json.JSONDecodeError):
                triplet = list(val_exps[min(index, len(val_exps) - 1)])
            prefix = json.dumps(list(triplet), separators=(",", ":"))
            item = {
                "sample_id": prefix,
                "source": str(triplet[0]),
                "shape_reference": str(triplet[1]),
                "color_reference": str(triplet[2]),
            }
            payload = {
                "base_rgb": base[index:index + 1], "strong_anchor_rgb": anchor[index:index + 1],
                "carrier_rgb": aux["new_hair_rgb_v242_carrier"][index:index + 1],
                "target_hair_mask": runtime["coarse_target_hair_mask"][index:index + 1],
                "source_hair_mask": runtime["source_hair_mask"][index:index + 1],
                "source_face_mask": runtime["source_face_mask"][index:index + 1],
                "source_skin_mask": runtime["source_skin_mask"][index:index + 1],
                "reference_hair_mask": runtime["reference_hair_mask"][index:index + 1],
                "v244_alpha": aux["target_hair_alpha_final"][index:index + 1],
                "v244_final": final[index:index + 1],
                "anchor_hair_evidence": aux["anchor_hair_evidence"][index:index + 1],
                "strand_structure_confidence": aux["strand_structure_confidence"][index:index + 1],
                "distance_prior": aux["distance_prior"][index:index + 1],
                "parser_labels": prepared["v230_parser_labels"][index:index + 1],
                "carrier_lab": aux["carrier_lab"][index:index + 1],
                "desired_l_low": aux["desired_l_low"][index:index + 1],
                "reference_ab_low": aux["target_ab_low"][index:index + 1],
                "scene_illumination_ab": aux["scene_illumination_ab"][index:index + 1],
                "source_hair_l": aux.get("source_hair_l", torch.zeros(1, 1, 1, 1, device=device))[index:index + 1] if aux.get("source_hair_l") is not None else torch.zeros(1, 1, 1, 1, device=device),
                "reference_l": aux.get("l_q50", torch.zeros(1, 1, 1, 1, device=device))[index:index + 1],
                "color_reference_rgb": runtime["color_reference_rgb"][index:index + 1],
                "scene_illumination_reliability": aux.get("scene_illumination_reliability", torch.ones_like(aux["target_hair_alpha_final"]))[index:index + 1],
                "shadow_chroma_scale": aux["shadow_chroma_scale"][index:index + 1],
                "highlight_chroma_scale": aux["highlight_chroma_scale"][index:index + 1],
            }
            save_cache(cache_dir, prefix, payload)
            rows.append(item)
    write_manifest(manifest_path, rows)
    return rows


def _run_matte(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    from models.matte_probe_v245 import MatteProbeV245
    from models.v245_death_test_common import composite, load_cache
    from utils.v245_matte_metrics import aggregate_matte, matte_metric_tensors
    probe = MatteProbeV245()
    records = []
    for index, row in enumerate(rows):
        data = load_cache(args.output_root / "cache", row["sample_id"])
        m1, aux = probe(coarse_target_hair_mask=data["target_hair_mask"], distance_prior=data["distance_prior"], anchor_hair_evidence=data["anchor_hair_evidence"], strand_structure_confidence=data["strand_structure_confidence"], source_skin_mask=data["source_skin_mask"], strong_anchor_rgb=data["strong_anchor_rgb"], return_aux=True)
        variants = {"m0": data["v244_alpha"], "m1": m1, "m2": data["target_hair_mask"]}
        record = {"sample_id": row["sample_id"]}
        for name, alpha in variants.items():
            values = matte_metric_tensors(base_rgb=data["base_rgb"], carrier_rgb=data["carrier_rgb"], alpha=alpha, coarse_target_hair_mask=data["target_hair_mask"], source_face_mask=data["source_face_mask"], source_skin_mask=data["source_skin_mask"], hair_support=aux["hair_support"], independent_candidate=aux["independent_flyaway_candidate"], hair_core=data["target_hair_mask"])
            record.update({f"{name}_{key}": float(value[0].item()) for key, value in values.items()})
            torch.save({"alpha": alpha.cpu(), "final": (alpha * data["carrier_rgb"] + (1.0 - alpha) * data["base_rgb"]).cpu()}, args.output_root / "matte" / "alpha" / f"{index:03d}_{name}.pt")
        if index < 8:
            _save_preview(args.output_root / "matte" / "visual" / f"sample_{index:03d}.png", [data["strong_anchor_rgb"], data["carrier_rgb"], _mask_preview(data["target_hair_mask"]), _mask_preview(data["v244_alpha"]), composite(data["base_rgb"], data["carrier_rgb"], data["v244_alpha"]), _mask_preview(m1), composite(data["base_rgb"], data["carrier_rgb"], m1)])
        records.append(record)
    summary = aggregate_matte(records)
    _write_jsonl(args.output_root / "matte" / "per_sample.jsonl", records)
    (args.output_root / "matte" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_appearance(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    from models.appearance_probe_v245 import AppearanceProbeV245
    from models.v245_death_test_common import composite, load_cache
    from models.SG_IDCT_v16 import rgb_to_lab
    from utils.v245_appearance_metrics import appearance_metric_tensors, classify_appearance
    probe = AppearanceProbeV245()
    records = []
    for index, row in enumerate(rows):
        data = load_cache(args.output_root / "cache", row["sample_id"])
        trusted = probe.trusted_core(data["target_hair_mask"], data["anchor_hair_evidence"])
        outputs, aux = probe(carrier_lab=data["carrier_lab"], desired_l_low=data["desired_l_low"], reference_ab_low=data["reference_ab_low"], scene_illumination_ab=data["scene_illumination_ab"], source_hair_l=data["source_hair_l"], reference_l=data["reference_l"], base_rgb=data["base_rgb"], trusted_alpha=trusted, return_aux=True)
        outputs["a0_carrier_rgb"] = data["carrier_rgb"]
        record = {"sample_id": row["sample_id"]}
        for name, rgb in outputs.items():
            if not name.endswith("_rgb"):
                continue
            variant = name.split("_")[0]
            metrics = appearance_metric_tensors(carrier_rgb=data["carrier_rgb"], output_rgb=rgb, trusted_core=trusted, scene_illumination_ab=data["scene_illumination_ab"], scene_reliability=data["scene_illumination_reliability"], shadow_scale=aux["a1"]["shadow_chroma_scale"], highlight_scale=aux["a1"]["highlight_chroma_scale"])
            record.update({f"{variant}_{key}": float(value[0].item()) for key, value in metrics.items()})
            chroma = float(torch.linalg.vector_norm(rgb_to_lab(data["color_reference_rgb"])[:, 1:], dim=1).mean().item())
            category = "low_chroma" if chroma < 12.0 else "normal_chroma" if chroma < 25.0 else "high_chroma"
            torch.save(rgb.cpu(), args.output_root / "appearance" / "previews" / category / f"{index:03d}_{variant}.pt")
        if index < 8:
            _save_preview(args.output_root / "appearance" / "visual" / f"sample_{index:03d}.png", [data["strong_anchor_rgb"], data["carrier_rgb"], outputs["a1_photometric_rgb"], outputs["a2_no_scene_rgb"], outputs["a3_no_shading_rgb"], _mask_preview(trusted)])
        records.append(record)
    summary = classify_appearance(records)
    _write_jsonl(args.output_root / "appearance" / "per_sample.jsonl", records)
    (args.output_root / "appearance" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_occluder(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    from models.occluder_probe_v245 import OccluderProbeV245
    from models.v245_death_test_common import load_cache
    from utils.v245_occluder_metrics import classify_occluder, occluder_metric_tensors
    probe = OccluderProbeV245()
    records = []
    for index, row in enumerate(rows):
        data = load_cache(args.output_root / "cache", row["sample_id"])
        real = probe.real_mask(data["parser_labels"], data["base_rgb"])
        synthetic = torch.zeros_like(real)
        for value in probe.synthetic_masks(data["base_rgb"]).values():
            synthetic = torch.maximum(synthetic, value)
        occluder = probe.soften(torch.maximum(real, synthetic).clamp(0, 1))
        current, protected = probe.apply(base_rgb=data["base_rgb"], hair_rgb=data["carrier_rgb"], hair_alpha=data["v244_alpha"], occluder_alpha=occluder)
        metrics = occluder_metric_tensors(base_rgb=data["base_rgb"], current_rgb=current, protected_rgb=protected, occluder_alpha=occluder, hair_alpha=data["v244_alpha"], hair_alpha_reference=data["v244_alpha"], hair_mask=data["target_hair_mask"], real_mask=real, synthetic_mask=synthetic)
        record = {"sample_id": row["sample_id"], **{key: float(value[0].item()) for key, value in metrics.items()}}
        torch.save({"real": real.cpu(), "synthetic": synthetic.cpu(), "combined": occluder.cpu()}, args.output_root / "occluder" / "masks" / f"{index:03d}.pt")
        torch.save({name: value.cpu() for name, value in probe.synthetic_masks(data["base_rgb"]).items()}, args.output_root / "occluder" / "synthetic" / f"{index:03d}.pt")
        if index < 8:
            _save_preview(args.output_root / "occluder" / "visual" / f"sample_{index:03d}.png", [data["strong_anchor_rgb"], _mask_preview(data["v244_alpha"]), _mask_preview(occluder), current, protected, (current - protected).abs()])
        records.append(record)
    summary = classify_occluder(records)
    _write_jsonl(args.output_root / "occluder" / "per_sample.jsonl", records)
    (args.output_root / "occluder" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.experiment != "all" and (args.output_root / "manifest.json").exists() and not args.reuse_v244_cache:
        raise RuntimeError(
            "A V2.45 manifest already exists. Pass --reuse-v244-cache so all experiments use the same frozen samples/cache."
        )
    if args.experiment in ("matte", "appearance", "occluder") and args.reuse_v244_cache:
        if not (args.output_root / "manifest.json").exists():
            raise RuntimeError("--reuse-v244-cache requires the V2.45 common manifest/cache")
    bt, trainer, post_process, val_exps, device = _setup_runtime(args)
    rows = _collect_cache(args, bt, trainer, post_process, val_exps, device)
    for name, directories in {
        "matte": ("visual", "alpha", "debug", "control"),
        "appearance": (
            "visual",
            "previews",
            "previews/low_chroma",
            "previews/normal_chroma",
            "previews/high_chroma",
            "debug",
        ),
        "occluder": ("visual", "masks", "synthetic", "debug"),
    }.items():
        for directory in directories:
            (args.output_root / name / directory).mkdir(parents=True, exist_ok=True)
    from utils.v245_death_test_summary import build_summary, write_summary
    results = {"matte": {"decision": "NOT_RUN"}, "appearance": {"decision": "NOT_RUN"}, "occluder": {"decision": "NOT_RUN"}}
    if args.experiment in ("matte", "all"):
        results["matte"] = _run_matte(args, rows)
    if args.experiment in ("appearance", "all"):
        results["appearance"] = _run_appearance(args, rows)
    if args.experiment in ("occluder", "all"):
        results["occluder"] = _run_occluder(args, rows)
    if args.experiment == "all":
        write_summary(args.output_root, build_summary(**results))
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
