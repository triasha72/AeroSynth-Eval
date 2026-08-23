import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aerosynth_eval.annotations import load_annotation_queue
from aerosynth_eval.contracts import DatasetSplit
from aerosynth_eval.mlx_vlm_runner import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MLX_VLM_MODEL,
    DEFAULT_TEMPERATURE,
    MlxVlmRunConfig,
)
from aerosynth_eval.vlm_batch_runner import (
    DevelopmentVlmBatchRunRecord,
    create_development_vlm_batch_plan,
    run_development_vlm_batch,
    summarize_development_vlm_batch,
    write_development_vlm_batch_record,
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
                    "rationale": "Test-double context score for batch orchestration.",
                },
                {
                    "dimension": "condition_fidelity",
                    "score": 3,
                    "rationale": "Test-double condition score for batch orchestration.",
                },
                {
                    "dimension": "image_quality",
                    "score": 3,
                    "rationale": "Test-double image-quality score for batch orchestration.",
                },
                {
                    "dimension": "inspection_utility",
                    "score": 3,
                    "rationale": "Test-double utility score for batch orchestration.",
                },
            ],
            "summary": "Synthetic test-double response for development batch testing.",
        }
    )


def _successful_inference(
    _: MlxVlmRunConfig,
    __: str,
    image_path: Path,
) -> str:
    return _valid_vlm_output(image_path)


@pytest.fixture(scope="module")
def successful_batch() -> DevelopmentVlmBatchRunRecord:
    return run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        inference=_successful_inference,
        executed_at=FIXED_TIME,
    )


def test_create_batch_plan_preserves_fixed_development_queue_order() -> None:
    queue = load_annotation_queue(QUEUE_PATH)

    plan = create_development_vlm_batch_plan(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
    )

    assert plan.inference_performed is False
    assert plan.performance_claim_supported is False
    assert plan.split is DatasetSplit.DEVELOPMENT
    assert len(plan.asset_ids) == 12
    assert len(plan.scenario_ids) == 12
    assert plan.asset_ids == tuple(record.asset_id for record in queue)
    assert plan.scenario_ids == tuple(record.scenario_id for record in queue)


def test_batch_records_all_successful_cases(
    successful_batch: DevelopmentVlmBatchRunRecord,
) -> None:
    assert successful_batch.inference_performed is True
    assert successful_batch.performance_claim_supported is False
    assert successful_batch.split is DatasetSplit.DEVELOPMENT
    assert len(successful_batch.cases) == 12
    assert all(case.status == "success" for case in successful_batch.cases)
    assert all(case.run_record is not None for case in successful_batch.cases)
    assert all(case.error is None for case in successful_batch.cases)


def test_batch_summary_reports_execution_reliability_only(
    successful_batch: DevelopmentVlmBatchRunRecord,
) -> None:
    summary = summarize_development_vlm_batch(successful_batch)

    assert summary.cases_attempted == 12
    assert summary.cases_succeeded == 12
    assert summary.cases_failed == 0
    assert summary.execution_success_rate == 1.0
    assert summary.performance_claim_supported is False


def test_batch_continues_after_one_invalid_model_response() -> None:
    failed_scenario = "wing-crack-close-diffuse"

    def partially_failing_inference(
        _: MlxVlmRunConfig,
        __: str,
        image_path: Path,
    ) -> str:
        if image_path.stem == failed_scenario:
            return "not-json"
        return _valid_vlm_output(image_path)

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        inference=partially_failing_inference,
        executed_at=FIXED_TIME,
    )

    summary = summarize_development_vlm_batch(record)
    failures = [case for case in record.cases if case.status == "failed"]

    assert summary.cases_attempted == 12
    assert summary.cases_succeeded == 11
    assert summary.cases_failed == 1
    assert len(failures) == 1
    assert failures[0].scenario_id == failed_scenario
    assert failures[0].run_record is None
    assert failures[0].error is not None
    assert "exactly one JSON object" in failures[0].error


def test_batch_bounds_oversized_failure_evidence() -> None:
    oversized_error = "session-load-failure-" + ("x" * 10_000)

    def oversized_failure(_: MlxVlmRunConfig) -> None:
        raise ValueError(oversized_error)

    record = run_development_vlm_batch(
        QUEUE_PATH,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        session_factory=oversized_failure,
        executed_at=FIXED_TIME,
    )

    assert all(case.status == "failed" for case in record.cases)
    assert all(case.error is not None for case in record.cases)
    assert all(len(case.error) == 2_000 for case in record.cases if case.error)
    assert all(
        re.search(r"\.\.\.\[truncated; sha256=[a-f0-9]{64}\]$", case.error)
        for case in record.cases
        if case.error
    )


def test_batch_plan_rejects_protected_test_split(tmp_path: Path) -> None:
    queue_text = QUEUE_PATH.read_text(encoding="utf-8")
    invalid_queue = tmp_path / "invalid_test_queue.csv"

    invalid_queue.write_text(
        queue_text.replace(",development,", ",test,", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protected test scenarios"):
        create_development_vlm_batch_plan(
            invalid_queue,
            _config(),
            REGISTRY_PATH,
            MATRIX_PATH,
            ASSET_ROOT,
        )


def test_write_batch_record_refuses_overwrite(
    tmp_path: Path,
    successful_batch: DevelopmentVlmBatchRunRecord,
) -> None:
    output_path = write_development_vlm_batch_record(successful_batch, tmp_path)

    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted["batch_id"] == successful_batch.batch_id
    assert len(persisted["cases"]) == 12
    assert persisted["performance_claim_supported"] is False

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_development_vlm_batch_record(successful_batch, tmp_path)
