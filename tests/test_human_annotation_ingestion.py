import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aerosynth_eval.annotations import load_annotation_queue
from aerosynth_eval.human_annotation_ingestion import (
    HUMAN_ALIGNMENT_STUDY_ID,
    RATER_TEMPLATE_COLUMNS,
    ingest_human_annotation_study,
    prepare_human_rater_template,
    summarize_human_annotation_ingestion,
    write_human_annotation_ingestion_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"
QUEUE_PATH = PROJECT_ROOT / "data" / "annotations" / "v0_1_development_annotation_queue.csv"

FIXED_TIME = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def _write_complete_submission(path: Path, rater_id: str, score: int) -> None:
    rows: list[dict[str, object]] = []

    for index, queue_record in enumerate(load_annotation_queue(QUEUE_PATH), start=1):
        rows.append(
            {
                "annotation_id": f"{rater_id}-{index:02d}",
                "rater_id": rater_id,
                "queue_id": queue_record.queue_id,
                "scenario_id": queue_record.scenario_id,
                "rubric_version": queue_record.rubric_version,
                "decision": "uncertain",
                "decision_rationale": "Software fixture: evidence remains intentionally cautious.",
                "context_fidelity": score,
                "context_fidelity_rationale": "Software fixture: context evidence is visible.",
                "condition_fidelity": score,
                "condition_fidelity_rationale": "Software fixture: condition evidence is visible.",
                "image_quality": score,
                "image_quality_rationale": "Software fixture: image quality is reviewable.",
                "inspection_utility": score,
                "inspection_utility_rationale": "Software fixture: research review only.",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RATER_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_human_rater_template_is_bound_to_full_queue(tmp_path: Path) -> None:
    output_path = tmp_path / "rater_alpha.csv"

    prepared = prepare_human_rater_template(
        "rater_alpha",
        output_path,
        QUEUE_PATH,
        REGISTRY_PATH,
        MATRIX_PATH,
    )

    with prepared.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    queue = load_annotation_queue(QUEUE_PATH)

    assert len(rows) == 12
    assert [row["scenario_id"] for row in rows] == [record.scenario_id for record in queue]
    assert {row["rater_id"] for row in rows} == {"rater_alpha"}
    assert all(row["decision"] == "" for row in rows)
    assert all(row["context_fidelity"] == "" for row in rows)


def test_prepare_template_refuses_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "rater_alpha.csv"

    prepare_human_rater_template(
        "rater_alpha",
        output_path,
        QUEUE_PATH,
        REGISTRY_PATH,
        MATRIX_PATH,
    )

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_human_rater_template(
            "rater_alpha",
            output_path,
            QUEUE_PATH,
            REGISTRY_PATH,
            MATRIX_PATH,
        )


def test_ingest_two_complete_distinct_submissions_without_agreement(tmp_path: Path) -> None:
    first = tmp_path / "rater_alpha.csv"
    second = tmp_path / "rater_beta.csv"

    _write_complete_submission(first, "rater_alpha", 2)
    _write_complete_submission(second, "rater_beta", 3)

    manifest = ingest_human_annotation_study(
        first,
        second,
        QUEUE_PATH,
        REGISTRY_PATH,
        MATRIX_PATH,
        executed_at=FIXED_TIME,
    )
    summary = summarize_human_annotation_ingestion(manifest)

    assert manifest.study_id == HUMAN_ALIGNMENT_STUDY_ID
    assert manifest.rater_count == 2
    assert manifest.records_per_rater == 12
    assert manifest.total_records == 24
    assert manifest.rater_ids == ("rater_alpha", "rater_beta")
    assert manifest.agreement_computed is False
    assert manifest.human_agreement_claim_supported is False
    assert manifest.human_authorship_verified_by_software is False
    assert manifest.independence_verified_by_software is False
    assert summary.total_records == 24
    assert summary.agreement_computed is False


def test_ingestion_rejects_same_rater_id(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    _write_complete_submission(first, "rater_alpha", 2)
    _write_complete_submission(second, "rater_alpha", 3)

    with pytest.raises(ValueError, match="two distinct rater IDs"):
        ingest_human_annotation_study(
            first,
            second,
            QUEUE_PATH,
            REGISTRY_PATH,
            MATRIX_PATH,
        )


def test_ingestion_rejects_synthetic_demo_rater_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    _write_complete_submission(first, "synthetic_demo_a", 2)
    _write_complete_submission(second, "rater_beta", 3)

    with pytest.raises(ValueError, match="fixture IDs"):
        ingest_human_annotation_study(
            first,
            second,
            QUEUE_PATH,
            REGISTRY_PATH,
            MATRIX_PATH,
        )


def test_ingestion_rejects_incomplete_submission(tmp_path: Path) -> None:
    first = tmp_path / "rater_alpha.csv"
    second = tmp_path / "rater_beta.csv"

    _write_complete_submission(first, "rater_alpha", 2)
    _write_complete_submission(second, "rater_beta", 3)

    with second.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    with second.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RATER_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="must annotate every scenario"):
        ingest_human_annotation_study(
            first,
            second,
            QUEUE_PATH,
            REGISTRY_PATH,
            MATRIX_PATH,
        )


def test_write_ingestion_manifest_persists_no_rating_content(tmp_path: Path) -> None:
    first = tmp_path / "rater_alpha.csv"
    second = tmp_path / "rater_beta.csv"

    _write_complete_submission(first, "rater_alpha", 2)
    _write_complete_submission(second, "rater_beta", 3)

    manifest = ingest_human_annotation_study(
        first,
        second,
        QUEUE_PATH,
        REGISTRY_PATH,
        MATRIX_PATH,
        executed_at=FIXED_TIME,
    )

    output_path = write_human_annotation_ingestion_manifest(manifest, tmp_path / "manifests")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    serialized = json.dumps(payload)
    assert payload["total_records"] == 24
    assert "decision_rationale" not in serialized
    assert "context_fidelity_rationale" not in serialized

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_human_annotation_ingestion_manifest(manifest, tmp_path / "manifests")
