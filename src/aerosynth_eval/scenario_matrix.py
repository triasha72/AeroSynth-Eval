"""Validation utilities for the frozen synthetic scenario design."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosynth_eval.contracts import AircraftRegion, DatasetSplit, SurfaceCondition

MATRIX_COLUMNS = frozenset(
    {
        "scenario_id",
        "split",
        "component",
        "condition",
        "capture_profile",
        "viewpoint",
        "lighting",
        "quality_challenge",
        "expected_evidence_visibility",
        "design_note",
    }
)
EXPECTED_CAPTURE_PROFILES = frozenset(
    {"close_diffuse", "oblique_directional", "close_glare", "oblique_blur"}
)


class ScenarioMatrixRecord(BaseModel):
    """One image-generation scenario in the v0.1 benchmark design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: DatasetSplit
    component: AircraftRegion
    condition: SurfaceCondition
    capture_profile: str = Field(min_length=3, max_length=64)
    viewpoint: str = Field(min_length=3, max_length=120)
    lighting: str = Field(min_length=3, max_length=120)
    quality_challenge: str = Field(min_length=2, max_length=64)
    expected_evidence_visibility: str = Field(min_length=3, max_length=64)
    design_note: str = Field(min_length=3, max_length=500)


class ScenarioMatrixSummary(BaseModel):
    """Balanced-design summary emitted after matrix validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    splits: dict[DatasetSplit, int]
    components: dict[AircraftRegion, int]
    conditions: dict[SurfaceCondition, int]
    capture_profiles: dict[str, int]
    quality_challenges: dict[str, int]


def load_scenario_matrix(path: Path) -> tuple[ScenarioMatrixRecord, ...]:
    """Load a CSV scenario matrix and reject malformed or duplicate records."""

    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                headers = next(reader)
            except StopIteration as error:
                raise ValueError(f"{path}: scenario matrix is empty.") from error

            if len(headers) != len(set(headers)) or set(headers) != MATRIX_COLUMNS:
                raise ValueError(
                    f"{path}: expected exactly these columns: {', '.join(sorted(MATRIX_COLUMNS))}."
                )

            records: list[ScenarioMatrixRecord] = []
            seen_ids: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                if not any(value.strip() for value in row):
                    continue
                if len(row) != len(headers):
                    raise ValueError(f"{path}:{line_number}: expected {len(headers)} CSV fields.")
                payload = {
                    header: value.strip() for header, value in zip(headers, row, strict=True)
                }
                try:
                    record = ScenarioMatrixRecord.model_validate(payload)
                except ValidationError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid scenario record: {error}"
                    ) from error
                if record.scenario_id in seen_ids:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate scenario_id '{record.scenario_id}'."
                    )
                seen_ids.add(record.scenario_id)
                records.append(record)
    except OSError as error:
        raise ValueError(f"Could not read scenario matrix at {path}.") from error

    if not records:
        raise ValueError(f"{path}: scenario matrix contains no records.")
    return tuple(records)


def summarize_scenario_matrix(
    records: tuple[ScenarioMatrixRecord, ...],
) -> ScenarioMatrixSummary:
    """Summarize the factor composition of a scenario matrix."""

    if not records:
        raise ValueError("Cannot summarize an empty scenario matrix.")

    return ScenarioMatrixSummary(
        record_count=len(records),
        splits={split: sum(record.split is split for record in records) for split in DatasetSplit},
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
        quality_challenges={
            challenge: sum(record.quality_challenge == challenge for record in records)
            for challenge in sorted({record.quality_challenge for record in records})
        },
    )


def validate_scenario_matrix(
    records: tuple[ScenarioMatrixRecord, ...],
) -> ScenarioMatrixSummary:
    """Enforce the frozen v0.1 balanced design and protected test split."""

    summary = summarize_scenario_matrix(records)
    if summary.record_count != 48:
        raise ValueError("v0.1 scenario matrix must contain exactly 48 records.")
    if summary.splits != {DatasetSplit.DEVELOPMENT: 36, DatasetSplit.TEST: 12}:
        raise ValueError("v0.1 scenario matrix must contain 36 development and 12 test records.")
    if set(summary.capture_profiles) != EXPECTED_CAPTURE_PROFILES:
        raise ValueError("v0.1 scenario matrix must use the four approved capture profiles.")
    if any(count != 12 for count in summary.capture_profiles.values()):
        raise ValueError("Each capture profile must occur exactly 12 times.")

    for component in AircraftRegion:
        if summary.components[component] != 16:
            raise ValueError(f"Component '{component}' must occur exactly 16 times.")
    for condition in SurfaceCondition:
        if summary.conditions[condition] != 12:
            raise ValueError(f"Condition '{condition}' must occur exactly 12 times.")
    for component in AircraftRegion:
        for condition in SurfaceCondition:
            pair = tuple(
                record
                for record in records
                if record.component is component and record.condition is condition
            )
            if len(pair) != 4:
                raise ValueError(f"{component}/{condition} must contain exactly four scenarios.")
            if sum(record.split is DatasetSplit.TEST for record in pair) != 1:
                raise ValueError(
                    f"{component}/{condition} must contain exactly one held-out test scenario."
                )
    return summary
