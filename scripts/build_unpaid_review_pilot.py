#!/usr/bin/env python3
"""Create two blank volunteer-review CSVs from the checked-in response template."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

OUTPUT_COLUMNS = (
    "annotation_id",
    "rater_id",
    "queue_id",
    "scenario_id",
    "rubric_version",
    "decision",
    "decision_rationale",
    "context_fidelity",
    "context_fidelity_rationale",
    "condition_fidelity",
    "condition_fidelity_rationale",
    "image_quality",
    "image_quality_rationale",
    "inspection_utility",
    "inspection_utility_rationale",
)


def build_pilot(source: Path, output_dir: Path) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        raise ValueError("The development annotation queue must contain rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    for reviewer_id in ("volunteer_a", "volunteer_b"):
        target = output_dir / f"aerosynth_pilot_{reviewer_id}.csv"
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for index, original in enumerate(rows, start=1):
                row = {column: "" for column in OUTPUT_COLUMNS}
                row.update(
                    annotation_id=f"{reviewer_id}-{index:02d}",
                    rater_id=reviewer_id,
                    queue_id=original["queue_id"],
                    scenario_id=original["scenario_id"],
                    rubric_version=original["rubric_version"],
                )
                writer.writerow(row)
    manifest = {
        "schema_version": "1.0",
        "purpose": "unpaid_volunteer_pilot",
        "cases": len(rows),
        "reviewers_required": 2,
        "compensation": "none",
        "human_reviews_completed": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/annotations/v0_1_development_annotation_queue.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_pilot(args.source, args.output_dir)


if __name__ == "__main__":
    main()
