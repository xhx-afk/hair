"""Regenerate the unified V2.45 death-test report from experiment summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v245_death_test_summary import build_summary, write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("res/v245_death_tests"))
    args = parser.parse_args()
    results = {}
    for name in ("matte", "appearance", "occluder"):
        path = args.root / name / "summary.json"
        results[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"decision": "NOT_RUN", "count": 0, "key_metrics": {}}
    write_summary(args.root, build_summary(**results))
    print(json.dumps(build_summary(**results), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
