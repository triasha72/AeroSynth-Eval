"""Experiment registry and statistically disciplined evaluator hillclimbing."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.preference_benchmark import BenchmarkPartition


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    judge_id: str
    prompt_version: str
    partition: BenchmarkPartition
    adaptation_id: str | None = None
    calibration_id: str | None = None
    human_agreement: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    swap_consistency: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0, le=1.0)
    failure_rate: float = Field(ge=0.0, le=1.0)


def quality_score(result: ExperimentResult) -> float:
    """Transparent selection policy; weights must be frozen before final-test use."""
    return (
        0.40 * result.human_agreement
        + 0.25 * result.macro_f1
        + 0.20 * result.swap_consistency
        + 0.10 * (1.0 - result.ece)
        + 0.05 * (1.0 - result.failure_rate)
    )


def rank_experiments(
    results: list[ExperimentResult],
) -> list[ExperimentResult]:
    invalid = [
        result.experiment_id
        for result in results
        if result.partition is not BenchmarkPartition.VALIDATION
    ]

    if invalid:
        joined = ", ".join(sorted(invalid))
        raise ValueError(
            "Experiment selection is restricted to validation "
            f"results; invalid experiments: {joined}"
        )

    return sorted(
        results,
        key=quality_score,
        reverse=True,
    )


def append_experiment(path: Path, result: ExperimentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
