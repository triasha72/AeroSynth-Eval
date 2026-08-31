#!/usr/bin/env python3
"""Assess whether the real AGDD baseline is eligible for operational release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes must be between zero and a positive total")
    estimate = successes / total
    denominator = 1 + z**2 / total
    center = (estimate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(estimate * (1 - estimate) / total + z**2 / (4 * total**2))
    return (center - margin / denominator, center + margin / denominator)


def assess(baseline: dict[str, object], audit: dict[str, object]) -> dict[str, object]:
    validation_rows = int(baseline["rows"]["validation"])
    exact_successes = round(baseline["validation"]["exact_match_accuracy"] * validation_rows)
    interval = wilson_interval(exact_successes, validation_rows)
    checks = {
        "commercially_usable_license": {
            "value": baseline["dataset_license"],
            "required": "commercial use permitted",
            "passed": False,
        },
        "minimum_validation_pairs": {
            "value": validation_rows,
            "minimum": 200,
            "passed": validation_rows >= 200,
        },
        "minimum_macro_f1": {
            "value": baseline["validation"]["macro_f1"],
            "minimum": 0.8,
            "passed": baseline["validation"]["macro_f1"] >= 0.8,
        },
        "source_manifest_present": {
            "value": audit["manifest_sha256"],
            "passed": len(audit["manifest_sha256"]) == 64,
        },
    }
    return {
        "schema_version": "1.0",
        "policy": "agdd-operational-release-v1",
        "decision": "approved" if all(item["passed"] for item in checks.values()) else "rejected",
        "checks": checks,
        "uncertainty": {
            "exact_match_successes": exact_successes,
            "validation_pairs": validation_rows,
            "exact_match_wilson_95": {"lower": interval[0], "upper": interval[1]},
        },
        "permitted_use": "noncommercial research evaluation only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.baseline.read_text()), json.loads(args.audit.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
