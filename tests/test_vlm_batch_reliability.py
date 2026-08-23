import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aerosynth_eval.autograder import AutograderRequest
from aerosynth_eval.mlx_vlm_runner import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MLX_VLM_MODEL,
    DEFAULT_TEMPERATURE,
    MlxVlmFailureKind,
    MlxVlmRequestBoundInference,
    MlxVlmRunConfig,
    MlxVlmRunnerError,
)
from aerosynth_eval.vlm_batch_runner import (
    VlmBatchFailureKind,
    VlmBatchRetryPolicy,
    run_development_vlm_batch,
    summarize_development_vlm_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = PROJECT_ROOT / "data" / "annotations" / "v0_1_development_annotation_queue.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
ASSET_ROOT = PROJECT_ROOT / "data"

FIXED_TIME = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def _config() -> MlxVlmRunConfig:
    return MlxVlmRunConfig(
        model_id=DEFAULT_MLX_VLM_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )


def _valid_vlm_output(image_path: Path) -> str:
    scenario_id = image_path.stem

    return json.dumps(
        {
            "asset_id": f"asset-{scenario_id}",
            "scenario_id": scenario_id,
            "rubric_version": "v0.1",
            "result_source": "vlm_output",
            "decision": "uncertain",
            "confidence": 0.5,
            "scores": [
                {
                    "dimension": "context_fidelity",
                    "score": 3,
                    "rationale": "Reliability test-double context score.",
                },
                {
                    "dimension": "condition_fidelity",
                    "score": 3,
                    "rationale": "Reliability test-double condition score.",
                },
                {
                    "dimension": "image_quality",
                    "score": 3,
                    "rationale": "Reliability test-double image-quality score.",
                },
                {
                    "dimension": "inspection_utility",
                    "score": 3,
                    "rationale": "Reliability test-double utility score.",
                },
            ],
            "summary": "Synthetic response for reliability-layer testing.",
        }
    )


def test_shared_session_factory_is_created_once_for_twelve_cases() -> None:
    factory_calls = 0
    inference_calls = 0
    request_asset_ids: list[str] = []

    def fake_session_factory(_: MlxVlmRunConfig) -> MlxVlmRequestBoundInference:
        nonlocal factory_calls
        factory_calls += 1

        def inference(
            __: MlxVlmRunConfig,
            ___: str,
            image_path: Path,
            request: AutograderRequest,
        ) -> str:
            nonlocal inference_calls
            inference_calls += 1
            request_asset_ids.append(request.asset_id)
            assert request.asset_id == f"asset-{image_path.stem}"
            assert request.scenario_id == image_path.stem
            return _valid_vlm_output(image_path)

        return inference

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        session_factory=fake_session_factory,
        retry_policy=VlmBatchRetryPolicy(max_retries=0),
        executed_at=FIXED_TIME,
    )

    assert factory_calls == 1
    assert inference_calls == 12
    assert len(request_asset_ids) == 12
    assert record.session_reused is True
    assert record.request_bound_output_enabled is True
    assert all(case.status == "success" for case in record.cases)
    assert all(
        case.run_record is not None and case.run_record.run_backend == "shared_local_mlx_vlm"
        for case in record.cases
    )


def test_transformers_shared_backend_has_truthful_provenance() -> None:
    def fake_session_factory(_: MlxVlmRunConfig) -> MlxVlmRequestBoundInference:
        def inference(
            __: MlxVlmRunConfig,
            ___: str,
            image_path: Path,
            ____: AutograderRequest,
        ) -> str:
            return _valid_vlm_output(image_path)

        return inference

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        session_factory=fake_session_factory,
        session_backend="shared_transformers_vlm",
        retry_policy=VlmBatchRetryPolicy(max_retries=0),
        executed_at=FIXED_TIME,
    )

    assert all(case.run_record is not None for case in record.cases)
    assert all(
        case.run_record is not None
        and case.run_record.run_backend == "shared_transformers_vlm"
        and case.run_record.runner == "transformers"
        for case in record.cases
    )


def test_transient_inference_failure_is_retried_once() -> None:
    target = "wing-crack-close-diffuse"
    calls: dict[str, int] = {}

    def flaky_inference(
        _: MlxVlmRunConfig,
        __: str,
        image_path: Path,
    ) -> str:
        scenario_id = image_path.stem
        calls[scenario_id] = calls.get(scenario_id, 0) + 1

        if scenario_id == target and calls[scenario_id] == 1:
            raise MlxVlmRunnerError(
                MlxVlmFailureKind.INFERENCE_FAILURE,
                "Synthetic transient inference failure.",
            )

        return _valid_vlm_output(image_path)

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        inference=flaky_inference,
        retry_policy=VlmBatchRetryPolicy(max_retries=1),
        executed_at=FIXED_TIME,
    )

    target_case = next(case for case in record.cases if case.scenario_id == target)
    summary = summarize_development_vlm_batch(record)

    assert target_case.status == "success"
    assert target_case.attempt_count == 2
    assert calls[target] == 2
    assert summary.cases_succeeded == 12
    assert summary.total_attempts == 13
    assert summary.retry_attempts == 1


def test_json_parse_failure_is_not_retried_and_retains_output() -> None:
    target = "wing-crack-close-diffuse"
    calls: dict[str, int] = {}
    raw = "not-json"

    def invalid_json_inference(
        _: MlxVlmRunConfig,
        __: str,
        image_path: Path,
    ) -> str:
        scenario_id = image_path.stem
        calls[scenario_id] = calls.get(scenario_id, 0) + 1

        if scenario_id == target:
            return raw

        return _valid_vlm_output(image_path)

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        inference=invalid_json_inference,
        retry_policy=VlmBatchRetryPolicy(max_retries=3),
        executed_at=FIXED_TIME,
    )

    target_case = next(case for case in record.cases if case.scenario_id == target)
    summary = summarize_development_vlm_batch(record)

    assert target_case.status == "failed"
    assert target_case.failure_kind is VlmBatchFailureKind.JSON_PARSE_FAILURE
    assert target_case.attempt_count == 1
    assert target_case.rejected_output is not None
    assert target_case.rejected_output.preview == raw
    assert target_case.rejected_output.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert calls[target] == 1
    assert summary.retry_attempts == 0
    assert summary.rejected_outputs_retained == 1
    assert summary.failure_counts[VlmBatchFailureKind.JSON_PARSE_FAILURE] == 1


def test_request_binding_failure_retains_rejected_output() -> None:
    target = "wing-crack-close-diffuse"

    def wrong_identity_inference(
        _: MlxVlmRunConfig,
        __: str,
        image_path: Path,
    ) -> str:
        if image_path.stem != target:
            return _valid_vlm_output(image_path)

        payload = json.loads(_valid_vlm_output(image_path))
        payload["asset_id"] = target
        return json.dumps(payload)

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        inference=wrong_identity_inference,
        retry_policy=VlmBatchRetryPolicy(max_retries=0),
        executed_at=FIXED_TIME,
    )

    target_case = next(case for case in record.cases if case.scenario_id == target)
    summary = summarize_development_vlm_batch(record)

    assert target_case.status == "failed"
    assert target_case.failure_kind is VlmBatchFailureKind.REQUEST_BINDING_FAILURE
    assert target_case.rejected_output is not None
    assert target_case.rejected_output.char_count > 0
    assert summary.rejected_outputs_retained == 1


def test_model_load_failure_marks_all_cases_without_fake_attempts() -> None:
    def failing_session_factory(_: MlxVlmRunConfig) -> MlxVlmRequestBoundInference:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            "Synthetic model initialization failure.",
        )

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        session_factory=failing_session_factory,
        executed_at=FIXED_TIME,
    )
    summary = summarize_development_vlm_batch(record)

    assert record.inference_performed is False
    assert record.session_reused is False
    assert all(case.status == "failed" for case in record.cases)
    assert all(case.attempt_count == 0 for case in record.cases)
    assert all(case.failure_kind is VlmBatchFailureKind.MODEL_LOAD_FAILURE for case in record.cases)
    assert summary.cases_scheduled == 12
    assert summary.cases_attempted == 0
    assert summary.cases_failed == 12
    assert summary.rejected_outputs_retained == 0
    assert summary.failure_counts[VlmBatchFailureKind.MODEL_LOAD_FAILURE] == 12


def test_retry_policy_rejects_unbounded_retry_counts() -> None:
    with pytest.raises(ValueError):
        VlmBatchRetryPolicy(max_retries=4)
