#!/usr/bin/env python3
"""Freeze AGDD's official validation set into selection and protected-test subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "4b5daa92929934f30b1155033c3ce67b7701960f"
DEFAULT_SALT = "aerosynth-eval-agdd-protected-v1"


def partition_ids(sample_ids: list[str], salt: str = DEFAULT_SALT) -> tuple[list[str], list[str]]:
    if len(sample_ids) < 2:
        raise ValueError("At least two sample IDs are required.")
    def digest(sample_id: str) -> str:
        return hashlib.sha256(f"{salt}:{sample_id}".encode()).hexdigest()

    ranked = sorted(set(sample_ids), key=digest)
    midpoint = len(ranked) // 2
    return sorted(ranked[:midpoint]), sorted(ranked[midpoint:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    label_root = args.dataset / "data" / "labels" / "val"
    sample_ids = [path.stem for path in sorted(label_root.glob("*.txt"))]
    selection_ids, test_ids = partition_ids(sample_ids, args.salt)
    payload = {
        "claim_scope": "frozen_agdd_selection_protected_test_split",
        "dataset": "AGDD",
        "source_commit": SOURCE_COMMIT,
        "source_split": "val",
        "salt": args.salt,
        "selection_ids": selection_ids,
        "test_ids": test_ids,
        "selection_count": len(selection_ids),
        "test_count": len(test_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
