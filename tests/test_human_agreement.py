import csv
from pathlib import Path

from aerosynth_eval.human_agreement import summarize_human_agreement


def _write(path: Path, rater: str, delta: int = 0) -> None:
    fields = [
        "scenario_id",
        "decision",
        "context_fidelity",
        "condition_fidelity",
        "image_quality",
        "inspection_utility",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(3):
            writer.writerow(
                {
                    "scenario_id": f"s{i}",
                    "decision": "accept",
                    "context_fidelity": 4 - delta,
                    "condition_fidelity": 4,
                    "image_quality": 3,
                    "inspection_utility": 3,
                }
            )


def test_human_agreement(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, "a")
    _write(b, "b")
    summary = summarize_human_agreement(a, b)
    assert summary.decision_exact_agreement == 1.0
