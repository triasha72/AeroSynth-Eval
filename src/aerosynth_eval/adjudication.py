"""Manual adjudication workflow for development-only human disagreements."""

from __future__ import annotations

import csv
from pathlib import Path

ADJUDICATION_COLUMNS = (
    "scenario_id",
    "rater_a_decision",
    "rater_b_decision",
    "adjudicated_decision",
    "adjudication_rationale",
    "context_fidelity",
    "condition_fidelity",
    "image_quality",
    "inspection_utility",
)


def _load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["scenario_id"]: row for row in csv.DictReader(stream)}


def prepare_adjudication_queue(path_a: Path, path_b: Path, output_path: Path) -> Path:
    a = _load(path_a)
    b = _load(path_b)
    if set(a) != set(b):
        raise ValueError("Rater submissions do not cover the same scenarios.")
    rows: list[dict[str, str]] = []
    for scenario in sorted(a):
        if a[scenario] == b[scenario]:
            continue
        rows.append(
            {
                "scenario_id": scenario,
                "rater_a_decision": a[scenario]["decision"],
                "rater_b_decision": b[scenario]["decision"],
                "adjudicated_decision": "",
                "adjudication_rationale": "",
                "context_fidelity": "",
                "condition_fidelity": "",
                "image_quality": "",
                "inspection_utility": "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite '{output_path}'.")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ADJUDICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def validate_completed_adjudication(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for line_number, row in enumerate(rows, start=2):
        if row["adjudicated_decision"] not in {"accept", "reject", "uncertain"}:
            raise ValueError(f"{path}:{line_number}: adjudicated_decision is incomplete.")
        if not row["adjudication_rationale"].strip():
            raise ValueError(f"{path}:{line_number}: adjudication_rationale is required.")
        for key in (
            "context_fidelity",
            "condition_fidelity",
            "image_quality",
            "inspection_utility",
        ):
            try:
                score = int(row[key])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {key} must be an integer 0-4.") from error
            if score not in range(5):
                raise ValueError(f"{path}:{line_number}: {key} must be 0-4.")
    return len(rows)
