"""Explicit V2.44 carrier-preserving residual photometric validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V244_DIAGNOSTIC_ONLY", "1")
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")


def run_static_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    required = (
        root / "models" / "hair_local_recomposition_v844.py",
        root / "models" / "hair_photometric_residual_v844.py",
        root / "utils" / "v244_metrics.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"V2.44 static contract missing files: {missing}")
    recomposition = (root / "models" / "hair_local_recomposition_v844.py").read_text(encoding="utf-8")
    residual = (root / "models" / "hair_photometric_residual_v844.py").read_text(encoding="utf-8")
    train = (root / "scripts" / "blending_train_v8.py").read_text(encoding="utf-8")
    checks = {
        "v242_carrier_input": 'new_hair_rgb_v242_carrier' in recomposition,
        "bounded_l_residual": "max_low_l_delta=12.0" in recomposition and "delta_l_low" in residual,
        "carrier_ab_detail": "carrier_ab_detail" in residual,
        "hair_only_candidate": "candidate_rgb = hair_support * corrected_rgb" in recomposition,
        "total_gamut": "total_gamut_scale" in residual and "v244_metric_tensors" in train,
        "v244_entry": "USER_V244_DIAGNOSTIC_ONLY" in train and 'version = "v244"' in train,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V2.44 static contract failed: {failed}")
    print(f"[V2.44] static contract PASS: {', '.join(checks)}")


def main() -> None:
    run_static_contract()
    from scripts import blending_train_v8

    print("[V2.44] carrier-preserving residual photometric validator; diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
