"""Versioned provenance contracts for synthetic image assets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aerosynth_eval.contracts import AssetLifecycleStatus, DatasetSplit
from aerosynth_eval.scenario_matrix import (
    ScenarioMatrixRecord,
    load_scenario_matrix,
    validate_scenario_matrix,
)


class AssetRegistryRecord(BaseModel):
    """One planned or generated synthetic image asset with reproducibility metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: DatasetSplit
    lifecycle_status: AssetLifecycleStatus
    image_reference: str = Field(min_length=1, max_length=500)
    generator_name: str | None = Field(default=None, min_length=1, max_length=100)
    generator_model: str | None = Field(default=None, min_length=1, max_length=200)
    generator_version: str | None = Field(default=None, min_length=1, max_length=200)
    generation_prompt: str | None = Field(default=None, min_length=8, max_length=4_000)
    seed: int | None = Field(default=None, ge=0)
    generated_at: datetime | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_note: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_lifecycle_evidence(self) -> AssetRegistryRecord:
        """Require full provenance only after an image has been generated."""

        generation_evidence = {
            "generator_name": self.generator_name,
            "generator_model": self.generator_model,
            "generator_version": self.generator_version,
            "generation_prompt": self.generation_prompt,
            "seed": self.seed,
            "generated_at": self.generated_at,
            "image_sha256": self.image_sha256,
        }
        if self.lifecycle_status is AssetLifecycleStatus.PLANNED:
            populated = [name for name, value in generation_evidence.items() if value is not None]
            if populated:
                raise ValueError(
                    "planned asset records must omit generation evidence; "
                    f"found: {', '.join(populated)}."
                )
            return self

        missing = [name for name, value in generation_evidence.items() if value is None]
        if missing:
            raise ValueError(
                f"{self.lifecycle_status} asset records require generation evidence; "
                f"missing: {', '.join(missing)}."
            )
        if self.generated_at is not None and self.generated_at.tzinfo is None:
            raise ValueError("generated_at must include a UTC offset.")
        return self


class AssetRegistrySummary(BaseModel):
    """Compact inventory summary emitted after registry validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    splits: dict[DatasetSplit, int]
    lifecycle_statuses: dict[AssetLifecycleStatus, int]
    records_with_generation_evidence: int = Field(ge=0)


class MaterializedAssetSummary(BaseModel):
    """Integrity summary for a fully materialized synthetic image corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    generated_assets: int = Field(ge=1)
    sha256_verified_assets: int = Field(ge=1)
    total_image_bytes: int = Field(ge=1)


def load_asset_registry(path: Path) -> tuple[AssetRegistryRecord, ...]:
    """Load a JSON Lines asset registry and reject duplicate identities."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read asset registry at {path}.") from error

    records: list[AssetRegistryRecord] = []
    seen_asset_ids: set[str] = set()
    seen_scenario_ids: set[str] = set()
    seen_references: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        try:
            record = AssetRegistryRecord.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"{path}:{line_number}: invalid asset record: {error}") from error

        for value, seen_values, field_name in (
            (record.asset_id, seen_asset_ids, "asset_id"),
            (record.scenario_id, seen_scenario_ids, "scenario_id"),
            (record.image_reference, seen_references, "image_reference"),
        ):
            if value in seen_values:
                raise ValueError(f"{path}:{line_number}: duplicate {field_name} '{value}'.")
            seen_values.add(value)
        records.append(record)

    if not records:
        raise ValueError(f"{path}: asset registry contains no records.")
    return tuple(records)


def write_asset_registry(path: Path, records: tuple[AssetRegistryRecord, ...]) -> None:
    """Write validated registry records as deterministic JSON Lines."""

    if not records:
        raise ValueError("Cannot write an empty asset registry.")
    serialized = "\n".join(
        json.dumps(record.model_dump(mode="json"), separators=(",", ":")) for record in records
    )
    try:
        path.write_text(f"{serialized}\n", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not write asset registry at {path}.") from error


def summarize_asset_registry(
    records: tuple[AssetRegistryRecord, ...],
) -> AssetRegistrySummary:
    """Summarize lifecycle and split composition for an asset registry."""

    if not records:
        raise ValueError("Cannot summarize an empty asset registry.")

    return AssetRegistrySummary(
        record_count=len(records),
        splits={split: sum(record.split is split for record in records) for split in DatasetSplit},
        lifecycle_statuses={
            status: sum(record.lifecycle_status is status for record in records)
            for status in AssetLifecycleStatus
        },
        records_with_generation_evidence=sum(
            record.lifecycle_status is not AssetLifecycleStatus.PLANNED for record in records
        ),
    )


def validate_asset_registry(
    records: tuple[AssetRegistryRecord, ...],
    scenarios: tuple[ScenarioMatrixRecord, ...],
) -> AssetRegistrySummary:
    """Validate one-to-one scenario coverage and canonical asset references."""

    validate_scenario_matrix(scenarios)
    if len(records) != len(scenarios):
        raise ValueError(
            "asset registry must contain exactly one record for every scenario; "
            f"found {len(records)} assets for {len(scenarios)} scenarios."
        )

    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    registry_scenario_ids = {record.scenario_id for record in records}
    if registry_scenario_ids != set(scenarios_by_id):
        missing = sorted(set(scenarios_by_id) - registry_scenario_ids)
        unexpected = sorted(registry_scenario_ids - set(scenarios_by_id))
        details: list[str] = []
        if missing:
            details.append(f"missing scenarios: {', '.join(missing)}")
        if unexpected:
            details.append(f"unknown scenarios: {', '.join(unexpected)}")
        raise ValueError(
            "asset registry does not match scenario matrix; " + "; ".join(details) + "."
        )

    for record in records:
        scenario = scenarios_by_id[record.scenario_id]
        if record.split is not scenario.split:
            raise ValueError(
                f"asset '{record.asset_id}' split '{record.split}' does not match "
                f"scenario '{record.scenario_id}' split '{scenario.split}'."
            )
        expected_reference = f"assets/v0_1/{record.scenario_id}.png"
        if record.image_reference != expected_reference:
            raise ValueError(
                f"asset '{record.asset_id}' must use canonical reference '{expected_reference}'."
            )

    return summarize_asset_registry(records)


def load_and_validate_asset_registry(
    registry_path: Path,
    scenario_matrix_path: Path,
) -> AssetRegistrySummary:
    """Load both provenance inputs and validate their one-to-one relationship."""

    return validate_asset_registry(
        load_asset_registry(registry_path),
        load_scenario_matrix(scenario_matrix_path),
    )


def validate_materialized_assets(
    records: tuple[AssetRegistryRecord, ...],
    asset_root: Path,
) -> MaterializedAssetSummary:
    """Verify every generated image exists and matches its recorded SHA-256 digest."""

    not_generated = [
        record.asset_id
        for record in records
        if record.lifecycle_status is not AssetLifecycleStatus.GENERATED
    ]
    if not_generated:
        raise ValueError(
            "Materialized corpus requires every record to be generated; "
            f"found non-generated assets: {', '.join(not_generated)}."
        )

    total_image_bytes = 0
    for record in records:
        asset_path = asset_root / record.image_reference
        try:
            image_bytes = asset_path.read_bytes()
        except OSError as error:
            raise ValueError(f"Missing generated asset '{asset_path}'.") from error
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Generated asset '{asset_path}' is not a PNG file.")
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != record.image_sha256:
            raise ValueError(
                f"Generated asset '{asset_path}' does not match its recorded SHA-256 digest."
            )
        total_image_bytes += len(image_bytes)

    return MaterializedAssetSummary(
        record_count=len(records),
        generated_assets=len(records),
        sha256_verified_assets=len(records),
        total_image_bytes=total_image_bytes,
    )


def load_and_validate_materialized_corpus(
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
) -> MaterializedAssetSummary:
    """Validate frozen scenario coverage, registry provenance, and image integrity."""

    records = load_asset_registry(registry_path)
    scenarios = load_scenario_matrix(scenario_matrix_path)
    validate_asset_registry(records, scenarios)
    return validate_materialized_assets(records, asset_root)
