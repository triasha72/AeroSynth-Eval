#!/usr/bin/env python3
"""Run a frozen image-manifest augmentation comparison; never overwrite a result."""

import argparse
import hashlib
import json
from pathlib import Path

from aerosynth_eval.utility_experiment import run_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 67])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("result already exists; use a new experiment path")
    payload = a.manifest.read_bytes()
    result = run_experiment(json.loads(payload), a.data_root, budget=a.budget, seeds=a.seeds)
    result["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(result["status"])


if __name__ == "__main__":
    main()
