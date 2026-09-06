import csv
import json
import subprocess
import sys
from pathlib import Path


def test_builds_two_distinct_blank_rater_files(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_unpaid_review_pilot.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    with (tmp_path / "aerosynth_pilot_volunteer_a.csv").open(newline="") as stream:
        first = list(csv.DictReader(stream))
    with (tmp_path / "aerosynth_pilot_volunteer_b.csv").open(newline="") as stream:
        second = list(csv.DictReader(stream))
    assert len(first) == len(second) == 12
    assert {row["rater_id"] for row in first} == {"volunteer_a"}
    assert {row["rater_id"] for row in second} == {"volunteer_b"}
    assert all(not row["decision"] for row in first + second)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["human_reviews_completed"] is False
