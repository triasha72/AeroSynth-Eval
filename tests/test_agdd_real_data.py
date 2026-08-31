import json
from pathlib import Path

from scripts.assess_agdd_release import assess, wilson_interval
from scripts.train_agdd_real_baseline import classes_in


def test_classes_in_reads_multilabel_annotations(tmp_path: Path):
    path = tmp_path / "label.txt"
    path.write_text("2 0.1 0.2 0.3 0.4\n0 0.2 0.3 0.4 0.5\n")
    assert classes_in(path) == {0, 2}


def test_published_agdd_evidence_is_real_and_pinned():
    root = Path(__file__).parents[1]
    baseline = json.loads((root / "reports/agdd_real_baseline_v1.json").read_text())
    audit = json.loads((root / "reports/agdd_audit.json").read_text())
    assert baseline["contains_synthetic_data"] is False
    assert baseline["source_commit"] == audit["git_commit"]
    assert baseline["rows"] == {"train": 197, "validation": 22}
    assert audit["totals"]["paired_samples"] == 219
    assert len(audit["manifest_sha256"]) == 64


def test_release_assessment_rejects_current_noncommercial_small_baseline():
    root = Path(__file__).parents[1]
    baseline = json.loads((root / "reports/agdd_real_baseline_v1.json").read_text())
    audit = json.loads((root / "reports/agdd_audit.json").read_text())
    result = assess(baseline, audit)
    assert result["decision"] == "rejected"
    assert not result["checks"]["commercially_usable_license"]["passed"]
    lower, upper = wilson_interval(4, 22)
    assert lower < 4 / 22 < upper
