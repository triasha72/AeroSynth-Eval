#!/usr/bin/env python3
"""Write a content-free inventory receipt for the DLR dent archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.dlr_dent_archive import inspect_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = inspect_archive(str(args.archive))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
