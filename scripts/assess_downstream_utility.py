#!/usr/bin/env python3
"""Apply the downstream real-image utility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.downstream_utility import assess_downstream_utility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_downstream_utility(json.loads(args.experiment.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
