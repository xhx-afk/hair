"""Explicit V2.42 target-hair core ownership/new-growth validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V242_DIAGNOSTIC_ONLY", "1")
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")


def main() -> None:
    from scripts import blending_train_v8
    print("[V2.42] target-hair core ownership/new-growth diagnostic; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
