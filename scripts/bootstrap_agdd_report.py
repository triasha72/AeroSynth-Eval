#!/usr/bin/env python3
"""Add deterministic percentile-bootstrap intervals to an AGDD evaluation report."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def bootstrap_intervals(
    records: list[dict[str, object]], samples: int = 10_000, seed: int = 17
) -> dict[str, object]:
    if not records:
        raise ValueError("At least one evaluation record is required.")
    generator = random.Random(seed)
    accuracy_samples = []
    latency_samples = []
    for _ in range(samples):
        resample = [records[generator.randrange(len(records))] for _ in records]
        accuracy_samples.append(
            sum(bool(record["exact_match"]) for record in resample) / len(resample)
        )
        latency_samples.append(
            sum(float(record["latency_ms"]) for record in resample) / len(resample)
        )
    return {
        "method": "percentile bootstrap over cases",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "exact_match_accuracy": {
            "low": percentile(accuracy_samples, 0.025),
            "high": percentile(accuracy_samples, 0.975),
        },
        "mean_latency_ms": {
            "low": percentile(latency_samples, 0.025),
            "high": percentile(latency_samples, 0.975),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    payload = json.loads(args.report.read_text())
    result = {
        "claim_scope": "agdd_case_bootstrap_confidence_intervals",
        "source_report": str(args.report),
        "case_count": len(payload["records"]),
        "intervals": bootstrap_intervals(payload["records"], args.samples, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
