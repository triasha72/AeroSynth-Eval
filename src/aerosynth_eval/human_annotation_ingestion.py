"""Development-only ingestion workflow for pseudonymous human-rater submissions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aerosynth_eval.annotations import (
    RATER_ANNOTATION_COLUMNS,
    AnnotationQueueRecord,
    RaterAnnotationRecord,
    load_annotation_queue,
    load_rater_annotations,
    validate_annotation_queue,
    validate_rater_annotations,
)
from aerosynth_eval.asset_registry import load_asset_registry
from aerosynth_eval.contracts import DatasetSplit
from aerosynth_eval.scenario_matrix import load_scenario_matrix

HUMAN_ALIGNMENT_STUDY_ID = "v0_1_development_human_alignment"
SYNTHETIC_DEMO_RATER_PREFIX = "synthetic_demo_"

RATER_TEMPLATE_COLUMNS = (
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

if frozenset(RATER_TEMPLATE_COLUMNS) != RATER_ANNOTATION_COLUMNS:
    raise RuntimeError("Human-rater template columns disagree with the annotation contract.")


class HumanAnnotationSourceFingerprint(BaseModel):
    """Non-label provenance retained for one complete pseudonymous submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str = Field(min_length=1, max_length=255)
    rater_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,31}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    record_count: Literal[12] = 12
    scenario_count: Literal[12] = 12
    rubric_version: Literal["v0.1"] = "v0.1"


class HumanAnnotationIngestionManifest(BaseModel):
    """Provenance manifest proving two complete submissions were ingested."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["v0.1"] = "v0.1"
    manifest_id: str = Field(pattern=r"^human-ingest-[a-f0-9]{12}$")
    study_id: Literal["v0_1_development_human_alignment"] = "v0_1_development_human_alignment"
    data_classification: Literal["pseudonymous_human_annotation_inputs"] = (
        "pseudonymous_human_annotation_inputs"
    )
    claim_scope: Literal["ingestion_and_completeness_only"] = "ingestion_and_completeness_only"
    human_authorship_verified_by_software: Literal[False] = False
    independence_verified_by_software: Literal[False] = False
    agreement_computed: Literal[False] = False
    human_agreement_claim_supported: Literal[False] = False
    created_at: datetime
    queue_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    rubric_version: Literal["v0.1"] = "v0.1"
    queue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rater_count: Literal[2] = 2
    records_per_rater: Literal[12] = 12
    total_records: Literal[24] = 24
    rater_ids: tuple[str, str]
    sources: tuple[HumanAnnotationSourceFingerprint, HumanAnnotationSourceFingerprint]

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> Self:
        """Require two distinct pseudonymous raters and two distinct source files."""

        if self.split is not DatasetSplit.DEVELOPMENT:
            raise ValueError("Human annotation ingestion must remain development-only.")
        if self.rater_ids[0] == self.rater_ids[1]:
            raise ValueError("Human annotation ingestion requires two distinct rater IDs.")

        source_rater_ids = tuple(sorted(source.rater_id for source in self.sources))
        if source_rater_ids != tuple(sorted(self.rater_ids)):
            raise ValueError("Manifest rater IDs disagree with source fingerprints.")
        if self.sources[0].sha256 == self.sources[1].sha256:
            raise ValueError("Human annotation ingestion requires two distinct source files.")

        return self


class HumanAnnotationIngestionSummary(BaseModel):
    """Compact ingestion summary with no ratings or agreement statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_id: str
    study_id: str
    claim_scope: Literal["ingestion_and_completeness_only"] = "ingestion_and_completeness_only"
    queue_id: str
    split: DatasetSplit
    rubric_version: str
    rater_count: Literal[2] = 2
    records_per_rater: Literal[12] = 12
    total_records: Literal[24] = 24
    human_authorship_verified_by_software: Literal[False] = False
    independence_verified_by_software: Literal[False] = False
    agreement_computed: Literal[False] = False
    human_agreement_claim_supported: Literal[False] = False


def _validate_human_rater_id(rater_id: str) -> str:
    """Validate a pseudonymous rater ID without claiming real-world identity."""

    if re.fullmatch(r"[a-z][a-z0-9_-]{2,31}", rater_id) is None:
        raise ValueError(
            "rater_id must match ^[a-z][a-z0-9_-]{2,31}$ "
            "and should be a pseudonym rather than a person's name."
        )
    if rater_id.startswith(SYNTHETIC_DEMO_RATER_PREFIX):
        raise ValueError(
            f"Human annotation ingestion rejects '{SYNTHETIC_DEMO_RATER_PREFIX}' fixture IDs."
        )
    return rater_id


def _sha256_file(path: Path) -> str:
    """Hash one source file for reproducible ingestion provenance."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"Could not read annotation source file '{path}'.") from error


def _normalized_time(executed_at: datetime | None) -> datetime:
    """Normalize an optional timestamp to UTC."""

    timestamp = executed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("executed_at must include a UTC offset.")
    return timestamp.astimezone(UTC).replace(microsecond=0)


def _load_validated_queue(
    queue_path: Path,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> tuple[AnnotationQueueRecord, ...]:
    """Load the frozen development queue and validate all provenance links."""

    queue = load_annotation_queue(queue_path)
    validate_annotation_queue(
        queue,
        load_asset_registry(registry_path),
        load_scenario_matrix(scenario_matrix_path),
    )
    return queue


def prepare_human_rater_template(
    rater_id: str,
    output_path: Path,
    queue_path: Path,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> Path:
    """Create one blank private CSV template bound to the approved development queue."""

    normalized_rater_id = _validate_human_rater_id(rater_id)
    queue = _load_validated_queue(queue_path, registry_path, scenario_matrix_path)

    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing rater template '{output_path}'.")

    rows: list[dict[str, str]] = []
    for index, queue_record in enumerate(queue, start=1):
        rows.append(
            {
                "annotation_id": f"{normalized_rater_id}-{index:02d}",
                "rater_id": normalized_rater_id,
                "queue_id": queue_record.queue_id,
                "scenario_id": queue_record.scenario_id,
                "rubric_version": queue_record.rubric_version,
                "decision": "",
                "decision_rationale": "",
                "context_fidelity": "",
                "context_fidelity_rationale": "",
                "condition_fidelity": "",
                "condition_fidelity_rationale": "",
                "image_quality": "",
                "image_quality_rationale": "",
                "inspection_utility": "",
                "inspection_utility_rationale": "",
            }
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RATER_TEMPLATE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise ValueError(f"Could not write human-rater template '{output_path}'.") from error

    return output_path


def _load_single_complete_submission(
    path: Path,
    queue: tuple[AnnotationQueueRecord, ...],
) -> tuple[str, tuple[RaterAnnotationRecord, ...]]:
    """Load one complete file that contains exactly one non-fixture pseudonymous rater."""

    records = load_rater_annotations(path)
    summary = validate_rater_annotations(records, queue)

    if summary.rater_count != 1:
        raise ValueError(f"{path}: each ingestion source must contain exactly one rater_id.")

    rater_id = _validate_human_rater_id(records[0].rater_id)
    return rater_id, records


def _source_fingerprint(
    path: Path,
    rater_id: str,
    records: tuple[RaterAnnotationRecord, ...],
) -> HumanAnnotationSourceFingerprint:
    """Create non-label provenance for one validated source."""

    return HumanAnnotationSourceFingerprint(
        source_name=path.name,
        rater_id=rater_id,
        sha256=_sha256_file(path),
        record_count=12,
        scenario_count=12,
        rubric_version="v0.1",
    )


def ingest_human_annotation_study(
    annotations_a: Path,
    annotations_b: Path,
    queue_path: Path,
    registry_path: Path,
    scenario_matrix_path: Path,
    *,
    executed_at: datetime | None = None,
) -> HumanAnnotationIngestionManifest:
    """Validate and fingerprint two complete development human-rater submissions.

    The function validates software-level completeness and provenance only. It
    cannot verify that a person authored a file or that raters worked
    independently, and it intentionally computes no agreement statistics.
    """

    queue = _load_validated_queue(queue_path, registry_path, scenario_matrix_path)

    first_rater_id, first_records = _load_single_complete_submission(annotations_a, queue)
    second_rater_id, second_records = _load_single_complete_submission(annotations_b, queue)

    if first_rater_id == second_rater_id:
        raise ValueError("Human annotation ingestion requires two distinct rater IDs.")

    first_annotation_ids = {record.annotation_id for record in first_records}
    second_annotation_ids = {record.annotation_id for record in second_records}
    overlapping_annotation_ids = first_annotation_ids & second_annotation_ids
    if overlapping_annotation_ids:
        duplicate = sorted(overlapping_annotation_ids)[0]
        raise ValueError(
            f"Human annotation submissions reuse annotation_id '{duplicate}' across raters."
        )

    first_source = _source_fingerprint(annotations_a, first_rater_id, first_records)
    second_source = _source_fingerprint(annotations_b, second_rater_id, second_records)
    if first_source.sha256 == second_source.sha256:
        raise ValueError("Human annotation ingestion requires two distinct source files.")

    sorted_rater_ids = sorted((first_rater_id, second_rater_id))
    rater_ids = (sorted_rater_ids[0], sorted_rater_ids[1])
    sources_by_rater = {
        first_source.rater_id: first_source,
        second_source.rater_id: second_source,
    }
    sources = (
        sources_by_rater[rater_ids[0]],
        sources_by_rater[rater_ids[1]],
    )

    return HumanAnnotationIngestionManifest(
        manifest_id=f"human-ingest-{uuid.uuid4().hex[:12]}",
        created_at=_normalized_time(executed_at),
        queue_id=queue[0].queue_id,
        queue_sha256=_sha256_file(queue_path),
        rater_ids=rater_ids,
        sources=sources,
    )


def write_human_annotation_ingestion_manifest(
    manifest: HumanAnnotationIngestionManifest,
    output_directory: Path,
) -> Path:
    """Persist only ingestion provenance; do not copy human rating content."""

    output_path = output_directory / f"{manifest.manifest_id}.json"
    if output_path.exists():
        raise ValueError(
            f"Refusing to overwrite existing human annotation manifest '{output_path}'."
        )

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(
            f"Could not write human annotation ingestion manifest '{output_path}'."
        ) from error

    return output_path


def summarize_human_annotation_ingestion(
    manifest: HumanAnnotationIngestionManifest,
) -> HumanAnnotationIngestionSummary:
    """Return ingestion/completeness metadata without labels or agreement metrics."""

    return HumanAnnotationIngestionSummary(
        manifest_id=manifest.manifest_id,
        study_id=manifest.study_id,
        queue_id=manifest.queue_id,
        split=manifest.split,
        rubric_version=manifest.rubric_version,
    )
