#!/usr/bin/env python3
"""Fit per-frame-count binary thresholds using selection records only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def anomaly_score(row: dict[str, object]) -> float:
    if "anomaly_probability" in row:
        return float(row["anomaly_probability"])
    return 1 - float(row["confidence"])


def balanced_accuracy(records: list[dict[str, object]], threshold: float) -> float:
    recalls = []
    for label in ("good", "anomaly"):
        rows = [row for row in records if row["reference"] == label]
        correct = 0
        for row in rows:
            score = anomaly_score(row)
            prediction = "anomaly" if score >= threshold else "good"
            correct += prediction == label
        recalls.append(correct / len(rows))
    return sum(recalls) / len(recalls)


def fit(records: list[dict[str, object]]) -> dict[str, object]:
    output = {}
    for count in sorted({int(row["frame_count"]) for row in records}):
        rows = [row for row in records if int(row["frame_count"]) == count]
        scores = sorted({anomaly_score(row) for row in rows})
        candidates = [0.0, *[(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)], 1.0]
        quality, _, threshold = max(
            (balanced_accuracy(rows, candidate), -candidate, candidate) for candidate in candidates
        )
        output[str(count)] = {"threshold": threshold, "selection_balanced_accuracy": quality}
    return {"method": "selection-only balanced-accuracy threshold", "thresholds": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    payload = fit(json.loads(args.report.read_text())["records"])
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
