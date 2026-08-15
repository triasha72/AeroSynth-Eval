import csv
from pathlib import Path

import pytest

from aerosynth_eval.adjudication import prepare_adjudication_queue, validate_completed_adjudication


def _write(path: Path, decision: str) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_id", "decision"])
        writer.writeheader()
        writer.writerow({"scenario_id": "s1", "decision": decision})


def test_adjudication_queue_requires_manual_completion(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "adj.csv"
    _write(a, "accept")
    _write(b, "reject")
    prepare_adjudication_queue(a, b, out)
    with pytest.raises(ValueError):
        validate_completed_adjudication(out)
