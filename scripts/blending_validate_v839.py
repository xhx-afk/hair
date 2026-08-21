"""Explicit V2.39 reference-hair intrinsic-tone validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V239_DIAGNOSTIC_ONLY", "1")
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.39] explicit validator; reference-hair intrinsic-tone calibration diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
