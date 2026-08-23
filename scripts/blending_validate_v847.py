"""Static V2.47 carrier-first validator; diagnostic only."""

from __future__ import annotations

from pathlib import Path
import importlib.util


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    residual_text = root.joinpath("models/hair_photometric_residual_v847.py").read_text(encoding="utf-8")
    probe_text = root.joinpath("models/appearance_probe_v247.py").read_text(encoding="utf-8")
    metrics_text = root.joinpath("utils/v247_appearance_metrics.py").read_text(encoding="utf-8")
    checks = {"error_estimator": root.joinpath("models/carrier_reference_error_v847.py").exists(), "reference_stats": root.joinpath("models/reference_ab_statistics_v847.py").exists(), "residual": root.joinpath("models/hair_photometric_residual_v847.py").exists(), "probe": root.joinpath("models/appearance_probe_v247.py").exists(), "noop_aux": root.joinpath("models/v247_noop_aux.py").exists(), "metrics": root.joinpath("utils/v247_appearance_metrics.py").exists(), "component_policy": root.joinpath("utils/v247_component_policy.py").exists(), "scene_tint_off": "scene_tint_enabled = False" in residual_text, "plausibility_off": "v247_plausibility_enabled = False" in residual_text, "plausibility_ablation": "plausibility_enabled=True" in probe_text, "hair_only_restore": "hair_apply * (rgb_safe - carrier_rgb)" in residual_text, "trusted_stats": "carrier_stats_mask" in residual_text and "stats_mask_source" in residual_text, "masked_quantile": "_masked_q" in residual_text, "batch_scalar": "def _batch_scalar" in metrics_text, "upper_tail": "def _upper_tail_mean" in metrics_text, "strict_non_hair": "strict_non_hair_max_rgb_change" in metrics_text, "selected_summary": "selected_summary.json" in root.joinpath("scripts/run_v247_appearance_death_test.py").read_text(encoding="utf-8"), "no_op_path": "build_carrier_noop_aux" in probe_text}
    print("[V2.47] carrier-first error-aware residual validator; diagnostic-only.")
    print("NOT integrated into formal Blending_v8 inference.")
    for name, passed in checks.items(): print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()): raise SystemExit(1)
    test_path = root / "scripts/test_v247_appearance.py"
    spec = importlib.util.spec_from_file_location("v247_correctness_tests", test_path)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module); module.run_checks()
    print("core_tests: PASS (18 checks)")


if __name__ == "__main__": main()
