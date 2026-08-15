"""Reliability-oriented development-only batch orchestration for VLM evaluation."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
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
from aerosynth_eval.evaluator_output import RejectedOutputProvenance
from aerosynth_eval.mlx_vlm_runner import (
    MlxVlmFailureKind,
    MlxVlmInference,
    MlxVlmRequestBoundInference,
    MlxVlmRunBackend,
    MlxVlmRunConfig,
    MlxVlmRunnerError,
    MlxVlmSessionFactory,
    MlxVlmSmokeRunRecord,
    create_mlx_vlm_session,
    run_mlx_vlm_smoke,
)
from aerosynth_eval.scenario_matrix import load_scenario_matrix

DEFAULT_BATCH_MAX_RETRIES = 1


class VlmBatchFailureKind(StrEnum):
    """Failure categories persisted by the development batch reliability layer."""

    MODEL_LOAD_FAILURE = "model_load_failure"
    INFERENCE_FAILURE = "inference_failure"
    EMPTY_RESPONSE = "empty_response"
    JSON_PARSE_FAILURE = "json_parse_failure"
    SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"
    REQUEST_BINDING_FAILURE = "request_binding_failure"
    PREPARATION_FAILURE = "preparation_failure"


_MLX_TO_BATCH_FAILURE: dict[MlxVlmFailureKind, VlmBatchFailureKind] = {
    MlxVlmFailureKind.MODEL_LOAD_FAILURE: VlmBatchFailureKind.MODEL_LOAD_FAILURE,
    MlxVlmFailureKind.INFERENCE_FAILURE: VlmBatchFailureKind.INFERENCE_FAILURE,
    MlxVlmFailureKind.EMPTY_RESPONSE: VlmBatchFailureKind.EMPTY_RESPONSE,
    MlxVlmFailureKind.JSON_PARSE_FAILURE: VlmBatchFailureKind.JSON_PARSE_FAILURE,
    MlxVlmFailureKind.SCHEMA_VALIDATION_FAILURE: VlmBatchFailureKind.SCHEMA_VALIDATION_FAILURE,
    MlxVlmFailureKind.REQUEST_BINDING_FAILURE: VlmBatchFailureKind.REQUEST_BINDING_FAILURE,
}


class VlmBatchRetryPolicy(BaseModel):
    """Bounded retry settings for transient development inference failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_retries: int = Field(default=DEFAULT_BATCH_MAX_RETRIES, ge=0, le=3)


class DevelopmentVlmBatchPlan(BaseModel):
    """Validated no-inference plan for the fixed development evaluation queue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["v0.3"] = "v0.3"
    batch_kind: Literal["development_vlm_batch"] = "development_vlm_batch"
    claim_scope: Literal["development_execution_reliability_only"] = (
        "development_execution_reliability_only"
    )
    performance_claim_supported: Literal[False] = False
    inference_performed: Literal[False] = False
    shared_session_enabled: Literal[True] = True
    request_bound_output_enabled: Literal[True] = True
    queue_id: str = Field(min_length=3, max_length=64)
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    asset_ids: tuple[str, ...] = Field(min_length=1)
    scenario_ids: tuple[str, ...] = Field(min_length=1)
    config: MlxVlmRunConfig
    retry_policy: VlmBatchRetryPolicy

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
    """Outcome and reliability provenance for one development batch case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(min_length=3, max_length=100)
    scenario_id: str = Field(min_length=3, max_length=100)
    status: Literal["success", "failed"]
    attempt_count: int = Field(ge=0, le=4)
    elapsed_seconds: float = Field(ge=0.0)
    failure_kind: VlmBatchFailureKind | None = None
    run_record: MlxVlmSmokeRunRecord | None = None
    error: str | None = Field(default=None, max_length=2_000)
    rejected_output: RejectedOutputProvenance | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require one internally consistent success or failure representation."""

        if self.status == "success":
            if self.attempt_count < 1:
                raise ValueError("Successful batch cases must record at least one attempt.")
            if self.run_record is None:
                raise ValueError("Successful batch cases must include a run record.")
            if (
                self.error is not None
                or self.failure_kind is not None
                or self.rejected_output is not None
            ):
                raise ValueError("Successful batch cases must not include failure metadata.")
            if self.run_record.request.asset_id != self.asset_id:
                raise ValueError("Batch case asset_id disagrees with its run record.")
            if self.run_record.request.scenario_id != self.scenario_id:
                raise ValueError("Batch case scenario_id disagrees with its run record.")
        else:
            if self.run_record is not None:
                raise ValueError("Failed batch cases must not include a run record.")
            if not self.error or self.failure_kind is None:
                raise ValueError("Failed batch cases must include typed failure metadata.")
            if self.attempt_count == 0 and (
                self.failure_kind is not VlmBatchFailureKind.MODEL_LOAD_FAILURE
            ):
                raise ValueError(
                    "Only model-load failures may occur before a case inference attempt."
                )

        return self


class DevelopmentVlmBatchRunRecord(BaseModel):
    """Complete reliability and provenance record for one development batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["v0.3"] = "v0.3"
    batch_id: str = Field(pattern=r"^vlm-batch-[a-f0-9]{12}$")
    batch_kind: Literal["development_vlm_batch"] = "development_vlm_batch"
    claim_scope: Literal["development_execution_reliability_only"] = (
        "development_execution_reliability_only"
    )
    performance_claim_supported: Literal[False] = False
    inference_performed: bool
    session_reused: bool
    request_bound_output_enabled: Literal[True] = True
    session_setup_seconds: float = Field(ge=0.0)
    batch_elapsed_seconds: float = Field(ge=0.0)
    executed_at: datetime
    queue_id: str = Field(min_length=3, max_length=64)
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    config: MlxVlmRunConfig
    retry_policy: VlmBatchRetryPolicy
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

        attempted = any(case.attempt_count > 0 for case in self.cases)
        if self.inference_performed != attempted:
            raise ValueError("Batch inference_performed disagrees with case attempt counts.")

        return self


class DevelopmentVlmBatchSummary(BaseModel):
    """Operational reliability summary without evaluator-quality claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    queue_id: str
    model_id: str
    split: DatasetSplit
    claim_scope: Literal["development_execution_reliability_only"] = (
        "development_execution_reliability_only"
    )
    performance_claim_supported: Literal[False] = False
    inference_performed: bool
    session_reused: bool
    request_bound_output_enabled: Literal[True] = True
    session_setup_seconds: float = Field(ge=0.0)
    batch_elapsed_seconds: float = Field(ge=0.0)
    cases_scheduled: int = Field(ge=1)
    cases_attempted: int = Field(ge=0)
    cases_succeeded: int = Field(ge=0)
    cases_failed: int = Field(ge=0)
    total_attempts: int = Field(ge=0)
    retry_attempts: int = Field(ge=0)
    rejected_outputs_retained: int = Field(ge=0)
    execution_success_rate: float = Field(ge=0.0, le=1.0)
    failure_counts: dict[VlmBatchFailureKind, int]


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


def _elapsed_seconds(started_at: float) -> float:
    """Return a stable non-negative elapsed duration for provenance."""

    return round(max(0.0, perf_counter() - started_at), 6)


def _batch_failure_kind(error: MlxVlmRunnerError) -> VlmBatchFailureKind:
    """Map typed single-run failures into persisted batch failure categories."""

    return _MLX_TO_BATCH_FAILURE[error.kind]


def _should_retry(
    failure_kind: VlmBatchFailureKind,
    attempt_count: int,
    retry_policy: VlmBatchRetryPolicy,
) -> bool:
    """Retry only bounded transient inference failures."""

    return (
        failure_kind is VlmBatchFailureKind.INFERENCE_FAILURE
        and attempt_count <= retry_policy.max_retries
    )


def create_development_vlm_batch_plan(
    queue_path: Path,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
    *,
    retry_policy: VlmBatchRetryPolicy | None = None,
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
    policy = retry_policy or VlmBatchRetryPolicy()

    return DevelopmentVlmBatchPlan(
        queue_id=queue[0].queue_id,
        asset_ids=tuple(record.asset_id for record in queue),
        scenario_ids=tuple(record.scenario_id for record in queue),
        config=config,
        retry_policy=policy,
    )


def _model_load_failure_record(
    queue: tuple[AnnotationQueueRecord, ...],
    config: MlxVlmRunConfig,
    retry_policy: VlmBatchRetryPolicy,
    executed_at: datetime | None,
    session_setup_seconds: float,
    batch_started_at: float,
    error: str,
) -> DevelopmentVlmBatchRunRecord:
    """Represent a global session-initialization failure without fake case attempts."""

    cases = tuple(
        VlmBatchCaseRecord(
            asset_id=queue_record.asset_id,
            scenario_id=queue_record.scenario_id,
            status="failed",
            attempt_count=0,
            elapsed_seconds=0.0,
            failure_kind=VlmBatchFailureKind.MODEL_LOAD_FAILURE,
            error=error,
        )
        for queue_record in queue
    )

    return DevelopmentVlmBatchRunRecord(
        batch_id=f"vlm-batch-{uuid.uuid4().hex[:12]}",
        inference_performed=False,
        session_reused=False,
        session_setup_seconds=session_setup_seconds,
        batch_elapsed_seconds=_elapsed_seconds(batch_started_at),
        executed_at=_normalized_execution_time(executed_at),
        queue_id=queue[0].queue_id,
        config=config,
        retry_policy=retry_policy,
        cases=cases,
    )


def run_development_vlm_batch(
    queue_path: Path,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
    *,
    inference: MlxVlmInference | None = None,
    session_factory: MlxVlmSessionFactory | None = None,
    retry_policy: VlmBatchRetryPolicy | None = None,
    executed_at: datetime | None = None,
) -> DevelopmentVlmBatchRunRecord:
    """Run the fixed queue with request-bound generation and typed reliability."""

    if inference is not None and session_factory is not None:
        raise ValueError("Provide either inference or session_factory, not both.")

    batch_started_at = perf_counter()
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
    policy = retry_policy or VlmBatchRetryPolicy()

    session_setup_seconds = 0.0
    session_reused = False
    inference_backend: MlxVlmRunBackend | None = None
    active_request_bound_inference: MlxVlmRequestBoundInference | None = None

    if inference is None:
        factory: MlxVlmSessionFactory = session_factory or create_mlx_vlm_session
        session_started_at = perf_counter()
        try:
            active_request_bound_inference = factory(config)
        except MlxVlmRunnerError as error:
            session_setup_seconds = _elapsed_seconds(session_started_at)
            return _model_load_failure_record(
                queue,
                config,
                policy,
                executed_at,
                session_setup_seconds,
                batch_started_at,
                str(error),
            )
        except ValueError as error:
            session_setup_seconds = _elapsed_seconds(session_started_at)
            return _model_load_failure_record(
                queue,
                config,
                policy,
                executed_at,
                session_setup_seconds,
                batch_started_at,
                str(error),
            )

        session_setup_seconds = _elapsed_seconds(session_started_at)
        session_reused = True
        inference_backend = "shared_local_mlx_vlm"

    cases: list[VlmBatchCaseRecord] = []

    for queue_record in queue:
        case_started_at = perf_counter()
        attempt_count = 0

        while True:
            attempt_count += 1
            try:
                run_record = run_mlx_vlm_smoke(
                    queue_record.asset_id,
                    config,
                    registry_path,
                    scenario_matrix_path,
                    asset_root,
                    inference=inference,
                    request_bound_inference=active_request_bound_inference,
                    inference_backend=inference_backend,
                    executed_at=executed_at,
                )
            except MlxVlmRunnerError as error:
                failure_kind = _batch_failure_kind(error)

                if _should_retry(
                    failure_kind,
                    attempt_count,
                    policy,
                ):
                    continue

                cases.append(
                    VlmBatchCaseRecord(
                        asset_id=queue_record.asset_id,
                        scenario_id=queue_record.scenario_id,
                        status="failed",
                        attempt_count=attempt_count,
                        elapsed_seconds=_elapsed_seconds(case_started_at),
                        failure_kind=failure_kind,
                        error=str(error),
                        rejected_output=error.rejected_output,
                    )
                )
                break
            except ValueError as error:
                cases.append(
                    VlmBatchCaseRecord(
                        asset_id=queue_record.asset_id,
                        scenario_id=queue_record.scenario_id,
                        status="failed",
                        attempt_count=attempt_count,
                        elapsed_seconds=_elapsed_seconds(case_started_at),
                        failure_kind=VlmBatchFailureKind.PREPARATION_FAILURE,
                        error=str(error),
                    )
                )
                break

            cases.append(
                VlmBatchCaseRecord(
                    asset_id=queue_record.asset_id,
                    scenario_id=queue_record.scenario_id,
                    status="success",
                    attempt_count=attempt_count,
                    elapsed_seconds=_elapsed_seconds(case_started_at),
                    run_record=run_record,
                )
            )
            break

    return DevelopmentVlmBatchRunRecord(
        batch_id=f"vlm-batch-{uuid.uuid4().hex[:12]}",
        inference_performed=any(case.attempt_count > 0 for case in cases),
        session_reused=session_reused,
        session_setup_seconds=session_setup_seconds,
        batch_elapsed_seconds=_elapsed_seconds(batch_started_at),
        executed_at=_normalized_execution_time(executed_at),
        queue_id=queue[0].queue_id,
        config=config,
        retry_policy=policy,
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

    scheduled = len(record.cases)
    attempted = sum(case.attempt_count > 0 for case in record.cases)
    succeeded = sum(case.status == "success" for case in record.cases)
    failed = scheduled - succeeded
    total_attempts = sum(case.attempt_count for case in record.cases)
    retry_attempts = sum(max(0, case.attempt_count - 1) for case in record.cases)
    rejected_outputs_retained = sum(case.rejected_output is not None for case in record.cases)

    failure_counts = {
        failure_kind: sum(case.failure_kind is failure_kind for case in record.cases)
        for failure_kind in VlmBatchFailureKind
    }

    return DevelopmentVlmBatchSummary(
        batch_id=record.batch_id,
        queue_id=record.queue_id,
        model_id=record.config.model_id,
        split=record.split,
        inference_performed=record.inference_performed,
        session_reused=record.session_reused,
        session_setup_seconds=record.session_setup_seconds,
        batch_elapsed_seconds=record.batch_elapsed_seconds,
        cases_scheduled=scheduled,
        cases_attempted=attempted,
        cases_succeeded=succeeded,
        cases_failed=failed,
        total_attempts=total_attempts,
        retry_attempts=retry_attempts,
        rejected_outputs_retained=rejected_outputs_retained,
        execution_success_rate=succeeded / scheduled,
        failure_counts=failure_counts,
    )
