"""Validation utilities for development-only human-rater annotation inputs."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosynth_eval.asset_registry import (
    AssetRegistryRecord,
    load_asset_registry,
    validate_asset_registry,
)
from aerosynth_eval.contracts import (
    AircraftRegion,
    DatasetSplit,
    EvaluationDecision,
    SurfaceCondition,
)
from aerosynth_eval.scenario_matrix import (
    EXPECTED_CAPTURE_PROFILES,
    ScenarioMatrixRecord,
    load_scenario_matrix,
)

DEVELOPMENT_QUEUE_ID = "v0_1_development_calibration"
RUBRIC_VERSION = "v0.1"

ANNOTATION_QUEUE_COLUMNS = frozenset(
    {
        "queue_id",
        "asset_id",
        "scenario_id",
        "split",
        "image_reference",
        "component",
        "condition",
        "capture_profile",
        "quality_challenge",
        "rubric_version",
    }
)
RATER_ANNOTATION_COLUMNS = frozenset(
    {
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
    }
)


class AnnotationQueueRecord(BaseModel):
    """One development-split asset assigned for independent human rating."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    queue_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: DatasetSplit
    image_reference: str = Field(min_length=1, max_length=500)
    component: AircraftRegion
    condition: SurfaceCondition
    capture_profile: str = Field(min_length=3, max_length=64)
    quality_challenge: str = Field(min_length=2, max_length=64)
    rubric_version: str = Field(min_length=1, max_length=32)


class RaterAnnotationRecord(BaseModel):
    """One rater's complete rubric response for one queued development asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    rater_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,31}$")
    queue_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    rubric_version: str = Field(min_length=1, max_length=32)
    decision: EvaluationDecision
    decision_rationale: str = Field(min_length=1, max_length=500)
    context_fidelity: int = Field(ge=0, le=4)
    context_fidelity_rationale: str = Field(min_length=1, max_length=500)
    condition_fidelity: int = Field(ge=0, le=4)
    condition_fidelity_rationale: str = Field(min_length=1, max_length=500)
    image_quality: int = Field(ge=0, le=4)
    image_quality_rationale: str = Field(min_length=1, max_length=500)
    inspection_utility: int = Field(ge=0, le=4)
    inspection_utility_rationale: str = Field(min_length=1, max_length=500)


class AnnotationQueueSummary(BaseModel):
    """Compact composition summary for a validated annotation queue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    queue_id: str
    record_count: int = Field(ge=1)
    split: DatasetSplit
    components: dict[AircraftRegion, int]
    conditions: dict[SurfaceCondition, int]
    capture_profiles: dict[str, int]


class RaterAnnotationSummary(BaseModel):
    """Completeness summary for one or more independent rater submissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    rater_count: int = Field(ge=1)
    annotations_per_rater: int = Field(ge=1)


def _load_csv_rows(
    path: Path,
    expected_columns: frozenset[str],
    record_label: str,
) -> tuple[tuple[int, dict[str, str]], ...]:
    """Read a strict CSV file while retaining source line numbers for errors."""

    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames
            if headers is None:
                raise ValueError(f"{path}: {record_label} CSV is empty.")
            normalized_headers = [header.strip() for header in headers]
            if (
                len(normalized_headers) != len(set(normalized_headers))
                or set(normalized_headers) != expected_columns
            ):
                raise ValueError(
                    f"{path}: expected exactly these columns: "
                    f"{', '.join(sorted(expected_columns))}."
                )

            rows: list[tuple[int, dict[str, str]]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"{path}:{line_number}: too many CSV fields.")
                normalized = {
                    key.strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                if not any(normalized.values()):
                    continue
                rows.append((line_number, normalized))
    except OSError as error:
        raise ValueError(f"Could not read {record_label} CSV at {path}.") from error

    if not rows:
        raise ValueError(f"{path}: {record_label} CSV contains no records.")
    return tuple(rows)


def load_annotation_queue(path: Path) -> tuple[AnnotationQueueRecord, ...]:
    """Load a strict CSV queue of development-split assets to be rated."""

    records: list[AnnotationQueueRecord] = []
    for line_number, payload in _load_csv_rows(path, ANNOTATION_QUEUE_COLUMNS, "annotation queue"):
        try:
            records.append(AnnotationQueueRecord.model_validate(payload))
        except ValidationError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid annotation queue record: {error}"
            ) from error
    return tuple(records)


def load_rater_annotations(path: Path) -> tuple[RaterAnnotationRecord, ...]:
    """Load a strict CSV submission from one or more pseudonymous human raters."""

    records: list[RaterAnnotationRecord] = []
    for line_number, payload in _load_csv_rows(path, RATER_ANNOTATION_COLUMNS, "rater annotation"):
        try:
            records.append(RaterAnnotationRecord.model_validate(payload))
        except ValidationError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid rater annotation record: {error}"
            ) from error
    return tuple(records)


def summarize_annotation_queue(
    records: tuple[AnnotationQueueRecord, ...],
) -> AnnotationQueueSummary:
    """Summarize the design balance of a non-empty annotation queue."""

    if not records:
        raise ValueError("Cannot summarize an empty annotation queue.")

    return AnnotationQueueSummary(
        queue_id=records[0].queue_id,
        record_count=len(records),
        split=records[0].split,
        components={
            component: sum(record.component is component for record in records)
            for component in AircraftRegion
        },
        conditions={
            condition: sum(record.condition is condition for record in records)
            for condition in SurfaceCondition
        },
        capture_profiles={
            profile: sum(record.capture_profile == profile for record in records)
            for profile in sorted({record.capture_profile for record in records})
        },
    )


def validate_annotation_queue(
    records: tuple[AnnotationQueueRecord, ...],
    assets: tuple[AssetRegistryRecord, ...],
    scenarios: tuple[ScenarioMatrixRecord, ...],
) -> AnnotationQueueSummary:
    """Require a balanced, development-only queue linked to frozen provenance."""

    validate_asset_registry(assets, scenarios)
    summary = summarize_annotation_queue(records)
    if summary.queue_id != DEVELOPMENT_QUEUE_ID:
        raise ValueError(f"Annotation queue must use queue_id '{DEVELOPMENT_QUEUE_ID}'.")
    if summary.record_count != 12:
        raise ValueError("Development annotation queue must contain exactly 12 records.")
    if summary.split is not DatasetSplit.DEVELOPMENT or any(
        record.split is not DatasetSplit.DEVELOPMENT for record in records
    ):
        raise ValueError("Development annotation queue must not contain protected test scenarios.")
    if {record.rubric_version for record in records} != {RUBRIC_VERSION}:
        raise ValueError(f"Annotation queue must use rubric_version '{RUBRIC_VERSION}'.")
    if len({record.scenario_id for record in records}) != len(records):
        raise ValueError("Annotation queue must not repeat scenario_id values.")
    if len({record.asset_id for record in records}) != len(records):
        raise ValueError("Annotation queue must not repeat asset_id values.")

    assets_by_id = {asset.asset_id: asset for asset in assets}
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for record in records:
        asset = assets_by_id.get(record.asset_id)
        if asset is None:
            raise ValueError(f"Annotation queue references unknown asset_id '{record.asset_id}'.")
        scenario = scenarios_by_id.get(record.scenario_id)
        if scenario is None:
            raise ValueError(
                f"Annotation queue references unknown scenario_id '{record.scenario_id}'."
            )
        if (
            record.scenario_id != asset.scenario_id
            or record.image_reference != asset.image_reference
            or record.split is not asset.split
        ):
            raise ValueError(
                f"Annotation queue record '{record.scenario_id}' disagrees with asset registry."
            )
        if (
            record.split is not scenario.split
            or record.component is not scenario.component
            or record.condition is not scenario.condition
            or record.capture_profile != scenario.capture_profile
            or record.quality_challenge != scenario.quality_challenge
        ):
            raise ValueError(
                f"Annotation queue record '{record.scenario_id}' disagrees with scenario matrix."
            )

    if any(count != 4 for count in summary.components.values()):
        raise ValueError(
            "Development annotation queue must contain four assets per aircraft region."
        )
    if any(count != 3 for count in summary.conditions.values()):
        raise ValueError(
            "Development annotation queue must contain three assets per surface condition."
        )
    if set(summary.capture_profiles) != EXPECTED_CAPTURE_PROFILES or any(
        count != 3 for count in summary.capture_profiles.values()
    ):
        raise ValueError(
            "Development annotation queue must contain three assets per capture profile."
        )
    return summary


def validate_rater_annotations(
    records: tuple[RaterAnnotationRecord, ...],
    queue: tuple[AnnotationQueueRecord, ...],
) -> RaterAnnotationSummary:
    """Require every submitted rater response to cover the full approved queue."""

    if not records:
        raise ValueError("Cannot validate an empty rater annotation submission.")
    if not queue:
        raise ValueError("Cannot validate rater annotations without an annotation queue.")

    queue_by_scenario = {record.scenario_id: record for record in queue}
    expected_scenarios = set(queue_by_scenario)
    seen_annotation_ids: set[str] = set()
    seen_rater_scenarios: set[tuple[str, str]] = set()
    for record in records:
        if record.annotation_id in seen_annotation_ids:
            raise ValueError(
                f"Rater annotations contain duplicate annotation_id '{record.annotation_id}'."
            )
        seen_annotation_ids.add(record.annotation_id)
        queue_record = queue_by_scenario.get(record.scenario_id)
        if queue_record is None:
            raise ValueError(
                f"Rater annotation references scenario_id '{record.scenario_id}' outside the queue."
            )
        if record.queue_id != queue_record.queue_id:
            raise ValueError(f"Rater annotation '{record.annotation_id}' has the wrong queue_id.")
        if record.rubric_version != queue_record.rubric_version:
            raise ValueError(
                f"Rater annotation '{record.annotation_id}' has the wrong rubric_version."
            )
        rater_scenario = (record.rater_id, record.scenario_id)
        if rater_scenario in seen_rater_scenarios:
            raise ValueError(
                f"Rater '{record.rater_id}' has multiple annotations for '{record.scenario_id}'."
            )
        seen_rater_scenarios.add(rater_scenario)

    rater_ids = sorted({record.rater_id for record in records})
    for rater_id in rater_ids:
        rater_scenarios = {record.scenario_id for record in records if record.rater_id == rater_id}
        if rater_scenarios != expected_scenarios:
            raise ValueError(
                f"Rater '{rater_id}' must annotate every scenario in the development queue."
            )

    return RaterAnnotationSummary(
        record_count=len(records),
        rater_count=len(rater_ids),
        annotations_per_rater=len(expected_scenarios),
    )


def load_and_validate_annotation_queue(
    queue_path: Path,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> AnnotationQueueSummary:
    """Load and validate the bundled queue against frozen scenario and asset inputs."""

    return validate_annotation_queue(
        load_annotation_queue(queue_path),
        load_asset_registry(registry_path),
        load_scenario_matrix(scenario_matrix_path),
    )


def load_and_validate_rater_annotations(
    annotation_path: Path,
    queue_path: Path,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> RaterAnnotationSummary:
    """Validate a complete rater submission against a validated development queue."""

    queue = load_annotation_queue(queue_path)
    validate_annotation_queue(
        queue,
        load_asset_registry(registry_path),
        load_scenario_matrix(scenario_matrix_path),
    )
    return validate_rater_annotations(load_rater_annotations(annotation_path), queue)
