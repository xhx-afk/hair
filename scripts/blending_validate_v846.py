"""Static V2.46 appearance-first validator; diagnostic only, no training."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = {
        "low_frequency_field": (root / "models/hair_low_frequency_illumination_v846.py").exists(),
        "carrier_safe_residual": (root / "models/hair_photometric_residual_v846.py").exists(),
        "appearance_probe": (root / "models/appearance_probe_v246.py").exists(),
        "scene_tint_disabled": "scene_tint_enabled = False" in (root / "models/hair_photometric_residual_v846.py").read_text(encoding="utf-8"),
        "appearance_metrics": (root / "utils/v246_appearance_metrics.py").exists(),
    }
    print("[V2.46] appearance-first carrier-safe photometric validator; diagnostic only; no training.")
    for name, passed in checks.items(): print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()): raise SystemExit(1)


if __name__ == "__main__":
    main()
