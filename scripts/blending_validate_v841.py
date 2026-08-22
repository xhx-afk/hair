"""Explicit V2.41 stable-hue/gamut/face-guard validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V241_DIAGNOSTIC_ONLY", "1")
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.41] explicit validator; stable hue + gamut solver + full face-overlap guard diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
