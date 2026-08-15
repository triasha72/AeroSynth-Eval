"""RichHF-18K normalized fine-grained human-feedback contracts and metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RichHFRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    example_id: str
    split: str
    filename: str
    aesthetics_score: float
    artifact_score: float
    misalignment_score: float
    overall_score: float


class RichHFPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    example_id: str
    aesthetics_score: float
    artifact_score: float
    misalignment_score: float
    overall_score: float


class DimensionMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    pearson_r: float = Field(ge=-1.0, le=1.0)


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def dimension_metric(human: list[float], predicted: list[float]) -> DimensionMetric:
    if len(human) != len(predicted) or not human:
        raise ValueError("Human and predicted vectors must be non-empty and equal length.")
    errors = [prediction - target for target, prediction in zip(human, predicted, strict=True)]
    return DimensionMetric(
        mae=sum(abs(error) for error in errors) / len(errors),
        rmse=math.sqrt(sum(error * error for error in errors) / len(errors)),
        pearson_r=_pearson(human, predicted),
    )


def load_richhf_jsonl(path: Path) -> list[RichHFRecord]:
    records: list[RichHFRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(RichHFRecord.model_validate_json(line))
    if not records:
        raise ValueError("No RichHF normalized records found.")
    return records


def write_normalized_records(records: list[RichHFRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Refusing to overwrite '{path}'.")
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
    return path
