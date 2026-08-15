"""Development-only batch orchestration for validated VLM evaluation runs."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aerosynth_eval.annotations import (
    AnnotationQueueRecord,
    load_annotation_queue,
    validate_annotation_queue,
)
from aerosynth_eval.asset_registry import (
    load_and_validate_materialized_corpus,
    load_asset_registry,
)
from aerosynth_eval.contracts import DatasetSplit
from aerosynth_eval.mlx_vlm_runner import (
    MlxVlmInference,
    MlxVlmRunConfig,
    MlxVlmSmokeRunRecord,
    run_mlx_vlm_smoke,
)
from aerosynth_eval.scenario_matrix import load_scenario_matrix


class DevelopmentVlmBatchPlan(BaseModel):
    """Validated no-inference plan for the fixed development evaluation queue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["v0.1"] = "v0.1"
    batch_kind: Literal["development_vlm_batch"] = "development_vlm_batch"
    claim_scope: Literal["development_execution_reliability_only"] = (
        "development_execution_reliability_only"
    )
    performance_claim_supported: Literal[False] = False
    inference_performed: Literal[False] = False
    queue_id: str = Field(min_length=3, max_length=64)
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    asset_ids: tuple[str, ...] = Field(min_length=1)
    scenario_ids: tuple[str, ...] = Field(min_length=1)
    config: MlxVlmRunConfig

    @model_validator(mode="after")
    def validate_development_scope(self) -> Self:
        """Keep the batch plan development-only and internally consistent."""

        if self.split is not DatasetSplit.DEVELOPMENT:
            raise ValueError("VLM batch plans must use only the development split.")
        if len(self.asset_ids) != len(self.scenario_ids):
            raise ValueError("Batch plan asset and scenario counts must match.")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("Batch plan must not contain duplicate asset IDs.")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Batch plan must not contain duplicate scenario IDs.")
        return self


class VlmBatchCaseRecord(BaseModel):
    """Outcome for one development asset attempted by the batch runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(min_length=3, max_length=100)
    scenario_id: str = Field(min_length=3, max_length=100)
    status: Literal["success", "failed"]
    run_record: MlxVlmSmokeRunRecord | None = None
    error: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require exactly one valid success or failure representation."""

        if self.status == "success":
            if self.run_record is None:
                raise ValueError("Successful batch cases must include a run record.")
            if self.error is not None:
                raise ValueError("Successful batch cases must not include an error.")
            if self.run_record.request.asset_id != self.asset_id:
                raise ValueError("Batch case asset_id disagrees with its run record.")
            if self.run_record.request.scenario_id != self.scenario_id:
                raise ValueError("Batch case scenario_id disagrees with its run record.")
        else:
            if self.run_record is not None:
                raise ValueError("Failed batch cases must not include a run record.")
            if not self.error:
                raise ValueError("Failed batch cases must include an error.")

        return self


class DevelopmentVlmBatchRunRecord(BaseModel):
    """Complete provenance record for one sequential development batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["v0.1"] = "v0.1"
    batch_id: str = Field(pattern=r"^vlm-batch-[a-f0-9]{12}$")
    batch_kind: Literal["development_vlm_batch"] = "development_vlm_batch"
    claim_scope: Literal["development_execution_reliability_only"] = (
        "development_execution_reliability_only"
    )
    performance_claim_supported: Literal[False] = False
    inference_performed: Literal[True] = True
    executed_at: datetime
    queue_id: str = Field(min_length=3, max_length=64)
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    config: MlxVlmRunConfig
    cases: tuple[VlmBatchCaseRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch_record(self) -> Self:
        """Keep persisted batch records unique and development-only."""

        if self.split is not DatasetSplit.DEVELOPMENT:
            raise ValueError("VLM batch records must use only the development split.")

        asset_ids = [case.asset_id for case in self.cases]
        scenario_ids = [case.scenario_id for case in self.cases]

        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("VLM batch record contains duplicate asset IDs.")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("VLM batch record contains duplicate scenario IDs.")

        return self


class DevelopmentVlmBatchSummary(BaseModel):
    """Operational summary without model-quality claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    queue_id: str
    model_id: str
    split: DatasetSplit
    claim_scope: Literal["development_execution_reliability_only"] = (
        "development_execution_reliability_only"
    )
    performance_claim_supported: Literal[False] = False
    inference_performed: Literal[True] = True
    cases_attempted: int = Field(ge=1)
    cases_succeeded: int = Field(ge=0)
    cases_failed: int = Field(ge=0)
    execution_success_rate: float = Field(ge=0.0, le=1.0)


def _load_validated_development_queue(
    queue_path: Path,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> tuple[AnnotationQueueRecord, ...]:
    """Load the fixed queue and enforce its existing development-only contract."""

    queue = load_annotation_queue(queue_path)
    validate_annotation_queue(
        queue,
        load_asset_registry(registry_path),
        load_scenario_matrix(scenario_matrix_path),
    )
    return queue


def _normalized_execution_time(executed_at: datetime | None) -> datetime:
    """Normalize an optional execution timestamp to UTC."""

    timestamp = executed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("executed_at must include a UTC offset.")
    return timestamp.astimezone(UTC).replace(microsecond=0)


def create_development_vlm_batch_plan(
    queue_path: Path,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
) -> DevelopmentVlmBatchPlan:
    """Validate the complete development batch without running inference."""

    queue = _load_validated_development_queue(
        queue_path,
        registry_path,
        scenario_matrix_path,
    )
    load_and_validate_materialized_corpus(
        registry_path,
        scenario_matrix_path,
        asset_root,
    )

    return DevelopmentVlmBatchPlan(
        queue_id=queue[0].queue_id,
        asset_ids=tuple(record.asset_id for record in queue),
        scenario_ids=tuple(record.scenario_id for record in queue),
        config=config,
    )


def run_development_vlm_batch(
    queue_path: Path,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
    *,
    inference: MlxVlmInference | None = None,
    executed_at: datetime | None = None,
) -> DevelopmentVlmBatchRunRecord:
    """Run every validated queue asset while isolating individual case failures."""

    queue = _load_validated_development_queue(
        queue_path,
        registry_path,
        scenario_matrix_path,
    )
    load_and_validate_materialized_corpus(
        registry_path,
        scenario_matrix_path,
        asset_root,
    )

    cases: list[VlmBatchCaseRecord] = []

    for queue_record in queue:
        try:
            run_record = run_mlx_vlm_smoke(
                queue_record.asset_id,
                config,
                registry_path,
                scenario_matrix_path,
                asset_root,
                inference=inference,
                executed_at=executed_at,
            )
        except ValueError as error:
            cases.append(
                VlmBatchCaseRecord(
                    asset_id=queue_record.asset_id,
                    scenario_id=queue_record.scenario_id,
                    status="failed",
                    error=str(error),
                )
            )
            continue

        cases.append(
            VlmBatchCaseRecord(
                asset_id=queue_record.asset_id,
                scenario_id=queue_record.scenario_id,
                status="success",
                run_record=run_record,
            )
        )

    return DevelopmentVlmBatchRunRecord(
        batch_id=f"vlm-batch-{uuid.uuid4().hex[:12]}",
        executed_at=_normalized_execution_time(executed_at),
        queue_id=queue[0].queue_id,
        config=config,
        cases=tuple(cases),
    )


def write_development_vlm_batch_record(
    record: DevelopmentVlmBatchRunRecord,
    output_directory: Path,
) -> Path:
    """Persist one complete batch without overwriting earlier results."""

    output_path = output_directory / f"{record.batch_id}.json"

    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing batch record '{output_path}'.")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(f"Could not write VLM batch record to '{output_path}'.") from error

    return output_path


def summarize_development_vlm_batch(
    record: DevelopmentVlmBatchRunRecord,
) -> DevelopmentVlmBatchSummary:
    """Summarize execution reliability without evaluating model quality."""

    succeeded = sum(case.status == "success" for case in record.cases)
    attempted = len(record.cases)
    failed = attempted - succeeded

    return DevelopmentVlmBatchSummary(
        batch_id=record.batch_id,
        queue_id=record.queue_id,
        model_id=record.config.model_id,
        split=record.split,
        cases_attempted=attempted,
        cases_succeeded=succeeded,
        cases_failed=failed,
        execution_success_rate=succeeded / attempted,
    )
