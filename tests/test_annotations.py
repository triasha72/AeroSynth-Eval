import csv
from pathlib import Path

import pytest

from aerosynth_eval.annotations import (
    RATER_ANNOTATION_COLUMNS,
    load_annotation_queue,
    load_rater_annotations,
    validate_annotation_queue,
    validate_rater_annotations,
)
from aerosynth_eval.asset_registry import load_asset_registry
from aerosynth_eval.contracts import DatasetSplit
from aerosynth_eval.scenario_matrix import load_scenario_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"
QUEUE_PATH = PROJECT_ROOT / "data" / "annotations" / "v0_1_development_annotation_queue.csv"


def _write_complete_submission(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(load_annotation_queue(QUEUE_PATH), start=1):
        rows.append(
            {
                "annotation_id": f"rater-a-{index:02d}",
                "rater_id": "rater_a",
                "queue_id": record.queue_id,
                "scenario_id": record.scenario_id,
                "rubric_version": record.rubric_version,
                "decision": "uncertain",
                "decision_rationale": (
                    "Synthetic baseline requires cautious visible-evidence review."
                ),
                "context_fidelity": 2,
                "context_fidelity_rationale": "The intended aircraft region is partially visible.",
                "condition_fidelity": 2,
                "condition_fidelity_rationale": (
                    "The requested condition has limited visible evidence."
                ),
                "image_quality": 2,
                "image_quality_rationale": (
                    "The procedural image is legible with visual limitations."
                ),
                "inspection_utility": 1,
                "inspection_utility_rationale": (
                    "The synthetic baseline is not suitable for operational use."
                ),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(RATER_ANNOTATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_annotation_queue_is_balanced_and_development_only() -> None:
    summary = validate_annotation_queue(
        load_annotation_queue(QUEUE_PATH),
        load_asset_registry(REGISTRY_PATH),
        load_scenario_matrix(MATRIX_PATH),
    )

    assert summary.record_count == 12
    assert summary.split is DatasetSplit.DEVELOPMENT
    assert set(summary.components.values()) == {4}
    assert set(summary.conditions.values()) == {3}
    assert set(summary.capture_profiles.values()) == {3}


def test_annotation_queue_rejects_a_protected_test_record() -> None:
    queue = load_annotation_queue(QUEUE_PATH)
    invalid_queue = (
        queue[0].model_copy(update={"split": DatasetSplit.TEST}),
        *queue[1:],
    )

    with pytest.raises(ValueError, match="must not contain protected test scenarios"):
        validate_annotation_queue(
            invalid_queue,
            load_asset_registry(REGISTRY_PATH),
            load_scenario_matrix(MATRIX_PATH),
        )


def test_complete_rater_submission_validates_against_queue(tmp_path: Path) -> None:
    submission_path = tmp_path / "rater_a.csv"
    _write_complete_submission(submission_path)

    summary = validate_rater_annotations(
        load_rater_annotations(submission_path),
        load_annotation_queue(QUEUE_PATH),
    )

    assert summary.record_count == 12
    assert summary.rater_count == 1
    assert summary.annotations_per_rater == 12


def test_incomplete_rater_submission_is_rejected(tmp_path: Path) -> None:
    submission_path = tmp_path / "rater_a.csv"
    _write_complete_submission(submission_path)
    incomplete = load_rater_annotations(submission_path)[:-1]

    with pytest.raises(ValueError, match="must annotate every scenario"):
        validate_rater_annotations(incomplete, load_annotation_queue(QUEUE_PATH))
