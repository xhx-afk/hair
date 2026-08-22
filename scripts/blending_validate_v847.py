"""Static V2.47 carrier-first validator; diagnostic only."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    residual_text = root.joinpath("models/hair_photometric_residual_v847.py").read_text(encoding="utf-8")
    checks = {"error_estimator": root.joinpath("models/carrier_reference_error_v847.py").exists(), "reference_stats": root.joinpath("models/reference_ab_statistics_v847.py").exists(), "residual": root.joinpath("models/hair_photometric_residual_v847.py").exists(), "probe": root.joinpath("models/appearance_probe_v247.py").exists(), "metrics": root.joinpath("utils/v247_appearance_metrics.py").exists(), "scene_tint_off": "scene_tint_enabled = False" in residual_text, "plausibility_off": "v247_plausibility_enabled = False" in residual_text, "hair_only_restore": "hair_apply * (rgb_safe - carrier_rgb)" in residual_text, "no_op_path": "error[\"no_op\"]" in residual_text}
    print("[V2.47] carrier-first error-aware residual validator; diagnostic only; no training.")
    for name, passed in checks.items(): print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()): raise SystemExit(1)


if __name__ == "__main__": main()
