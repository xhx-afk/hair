from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("res/v2451_death_tests"))
    args = parser.parse_args()
    root = args.root
    payload = json.loads((root / "death_test_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"report: {root / 'DEATH_TEST_REPORT.md'}")


if __name__ == "__main__":
    main()
