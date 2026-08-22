"""Explicit V2.41.2 final correctness/evaluation-alignment validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V2412_DIAGNOSTIC_ONLY", "1")
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")


def run_static_contract() -> None:
    """Reject known pretrain/evaluation wiring regressions before imports."""
    root = Path(__file__).resolve().parents[1]
    recomposition = (root / "models" / "hair_local_recomposition_v841.py").read_text(encoding="utf-8")
    resolver = (root / "models" / "target_hair_occlusion_resolver_v8412.py").read_text(encoding="utf-8")
    metrics = (root / "utils" / "v241_metrics.py").read_text(encoding="utf-8")
    train = (root / "scripts" / "blending_train_v8.py").read_text(encoding="utf-8")
    checks = {
        "resolver_path": "TargetHairOcclusionResolverV8412" in recomposition,
        "no_illumination_double_count": "mapped_l_low + illumination_aux[\"target_l_residual\"]" not in recomposition,
        "valid_only_aggregate": "aggregate_v2412_records" in train and "effective_hue_metric_valid" in metrics,
        "low_chroma_sample_valid": "low_chroma_sample_valid" in metrics,
        "safe_l_contract": "final_l_safe" in (root / "models" / "texture_preserving_recolor_v841.py").read_text(encoding="utf-8"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V2.41.2 static contract failed: {failed}")
    print(f"[V2.41.2] static contract PASS: {', '.join(checks)}")

def main() -> None:
    run_static_contract()
    from scripts import blending_train_v8
    print("[V2.41.2] final correctness/evaluation-alignment validator; diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
