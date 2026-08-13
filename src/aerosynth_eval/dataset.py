"""Utilities for versioned JSON Lines evaluation manifests."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosynth_eval.contracts import (
    AnnotationStatus,
    DatasetRecord,
    DatasetSplit,
    ImageProvenance,
)


class ManifestSummary(BaseModel):
    """A compact summary emitted after a manifest passes validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    splits: dict[DatasetSplit, int]
    provenance: dict[ImageProvenance, int]
    annotation_status: dict[AnnotationStatus, int]


def load_manifest(path: Path) -> tuple[DatasetRecord, ...]:
    """Load, validate, and de-duplicate records from a JSON Lines manifest."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read manifest at {path}.") from error

    records: list[DatasetRecord] = []
    seen_example_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        try:
            record = DatasetRecord.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"{path}:{line_number}: invalid dataset record: {error}") from error
        if record.example_id in seen_example_ids:
            raise ValueError(f"{path}:{line_number}: duplicate example_id '{record.example_id}'.")
        seen_example_ids.add(record.example_id)
        records.append(record)

    if not records:
        raise ValueError(f"{path}: manifest contains no records.")
    return tuple(records)


def summarize_manifest(records: tuple[DatasetRecord, ...]) -> ManifestSummary:
    """Summarize the split, provenance, and annotation composition of records."""

    if not records:
        raise ValueError("Cannot summarize an empty manifest.")

    return ManifestSummary(
        record_count=len(records),
        splits={split: sum(record.split is split for record in records) for split in DatasetSplit},
        provenance={
            provenance: sum(record.provenance is provenance for record in records)
            for provenance in ImageProvenance
        },
        annotation_status={
            status: sum(record.annotation_status is status for record in records)
            for status in AnnotationStatus
        },
    )
