"""Explicit V2.43 confidence-core/strand-matte/photometric validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set these before importing blending_train_v8: that module resolves the
# active diagnostic branch and cache namespace at import time.
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V243_DIAGNOSTIC_ONLY", "1")
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")


def run_static_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    required = {
        "confidence_core": root / "models" / "hair_core_confidence_v843.py",
        "strand_structure": root / "models" / "hair_strand_structure_v843.py",
        "strand_matte": root / "models" / "strand_aware_hair_matte_v843.py",
        "face_guard": root / "models" / "face_hair_matte_guard_v843.py",
        "photometric": root / "models" / "hair_photometric_appearance_v843.py",
        "scene_illumination": root / "models" / "scene_illumination_estimator_v843.py",
        "new_hair_illumination": root / "models" / "new_hair_illumination_v843.py",
        "metrics": root / "utils" / "v243_metrics.py",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"V2.43 static contract missing files: {missing}")
    recomposition = (root / "models" / "hair_local_recomposition_v843.py").read_text(encoding="utf-8")
    train = (root / "scripts" / "blending_train_v8.py").read_text(encoding="utf-8")
    checks = {
        "v843_recomposition": "HairLocalRecompositionV843" in recomposition,
        "confidence_core_wired": "HairCoreConfidenceV843" in recomposition,
        "strand_matte_wired": "StrandAwareHairMatteV843" in recomposition,
        "v243_entry_wired": "USER_V243_DIAGNOSTIC_ONLY" in train and 'version = "v243"' in train,
        "v243_metrics_wired": "v243_metric_tensors" in train,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V2.43 static contract failed: {failed}")
    print(f"[V2.43] static contract PASS: {', '.join(checks)}")


def main() -> None:
    run_static_contract()
    from scripts import blending_train_v8

    print("[V2.43] confidence-core + strand-aware matte + photometric validator; diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
