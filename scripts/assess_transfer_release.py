#!/usr/bin/env python3
"""Assess whether synthetic augmentation can advance past research evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.transfer_release import assess_transfer_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--human-alignment", type=Path)
    args = parser.parse_args()
    human = None
    if args.human_alignment:
        human = json.loads(args.human_alignment.read_text())
    result = assess_transfer_release(json.loads(args.experiment.read_text()), human)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
