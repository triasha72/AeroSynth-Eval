import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aerosynth_eval.autograder import build_autograder_request
from aerosynth_eval.contracts import DatasetSplit
from aerosynth_eval.mlx_vlm_runner import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MLX_VLM_MODEL,
    DEFAULT_TEMPERATURE,
    MlxLogitsProcessor,
    MlxVlmFailureKind,
    MlxVlmRunConfig,
    MlxVlmRunnerError,
    _build_autograder_logits_processor,
    _build_request_bound_autograder_logits_processor,
    _extract_mlx_vlm_text,
    create_mlx_vlm_smoke_plan,
    run_mlx_vlm_smoke,
    summarize_mlx_vlm_smoke_run,
    write_mlx_vlm_smoke_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
ASSET_ROOT = PROJECT_ROOT / "data"
DEVELOPMENT_ASSET_ID = "asset-fuselage-corrosion-close-diffuse"
TEST_ASSET_ID = "asset-fuselage-clean-close-diffuse"


class _FakeGenerationResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = object()


def test_build_autograder_logits_processor_uses_response_schema() -> None:
    processor = _FakeProcessor()
    captured: dict[str, object] = {}

    def fake_logits_processor(tokens: Any, logits: Any) -> Any:
        return logits

    def fake_builder(
        tokenizer: Any,
        schema: dict[str, Any],
    ) -> MlxLogitsProcessor:
        captured["tokenizer"] = tokenizer
        captured["schema"] = schema
        return fake_logits_processor

    result = _build_autograder_logits_processor(
        processor,
        fake_builder,
    )

    assert result is fake_logits_processor
    assert captured["tokenizer"] is processor.tokenizer

    schema = captured["schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"


def test_request_bound_logits_processor_uses_exact_request_identity() -> None:
    processor = _FakeProcessor()
    captured: dict[str, object] = {}
    request = build_autograder_request(
        DEVELOPMENT_ASSET_ID,
        REGISTRY_PATH,
        MATRIX_PATH,
    )

    def fake_logits_processor(tokens: Any, logits: Any) -> Any:
        return logits

    def fake_builder(
        tokenizer: Any,
        schema: dict[str, Any],
    ) -> MlxLogitsProcessor:
        captured["tokenizer"] = tokenizer
        captured["schema"] = schema
        return fake_logits_processor

    result = _build_request_bound_autograder_logits_processor(
        processor,
        fake_builder,
        request,
    )

    assert result is fake_logits_processor
    assert captured["tokenizer"] is processor.tokenizer

    schema = captured["schema"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert properties["asset_id"]["enum"] == [request.asset_id]
    assert properties["scenario_id"]["enum"] == [request.scenario_id]
    assert properties["rubric_version"]["enum"] == [request.rubric_version]
    assert properties["result_source"]["enum"] == ["vlm_output"]


def test_extract_mlx_vlm_text_accepts_generation_result_shape() -> None:
    result = _FakeGenerationResult('{"result_source": "vlm_output"}')

    assert _extract_mlx_vlm_text(result) == '{"result_source": "vlm_output"}'


def test_extract_mlx_vlm_text_accepts_legacy_string_shape() -> None:
    output = '{"result_source": "vlm_output"}'

    assert _extract_mlx_vlm_text(output) == output


def test_extract_mlx_vlm_text_rejects_empty_generation_result() -> None:
    result = _FakeGenerationResult("")

    with pytest.raises(ValueError, match="no non-empty text output"):
        _extract_mlx_vlm_text(result)


def _config() -> MlxVlmRunConfig:
    return MlxVlmRunConfig(
        model_id=DEFAULT_MLX_VLM_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )


def _valid_vlm_output() -> str:
    return json.dumps(
        {
            "asset_id": DEVELOPMENT_ASSET_ID,
            "scenario_id": "fuselage-corrosion-close-diffuse",
            "rubric_version": "v0.1",
            "result_source": "vlm_output",
            "decision": "uncertain",
            "confidence": 0.5,
            "scores": [
                {
                    "dimension": "context_fidelity",
                    "score": 3,
                    "rationale": "Test-double response: context is visible in the synthetic image.",
                },
                {
                    "dimension": "condition_fidelity",
                    "score": 2,
                    "rationale": "Test-double response: the condition is partially discernible.",
                },
                {
                    "dimension": "image_quality",
                    "score": 3,
                    "rationale": "Test-double response: the image is readable with limitations.",
                },
                {
                    "dimension": "inspection_utility",
                    "score": 2,
                    "rationale": "Test-double response: use remains limited to research review.",
                },
            ],
            "summary": (
                "Test-double response: exercises local-run provenance without model inference."
            ),
        }
    )


def _fake_inference(config: MlxVlmRunConfig, prompt: str, image_path: Path) -> str:
    assert config.model_id == DEFAULT_MLX_VLM_MODEL
    assert "result_source" in prompt
    assert image_path.is_file()
    return _valid_vlm_output()


def test_create_plan_preserves_development_only_provenance() -> None:
    plan = create_mlx_vlm_smoke_plan(
        DEVELOPMENT_ASSET_ID,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
    )

    assert plan.inference_performed is False
    assert plan.performance_claim_supported is False
    assert plan.request.split is DatasetSplit.DEVELOPMENT
    assert plan.request.asset_id == DEVELOPMENT_ASSET_ID
    assert len(plan.image_sha256) == 64
    assert len(plan.prompt_sha256) == 64


def test_create_plan_rejects_protected_test_asset() -> None:
    with pytest.raises(ValueError, match="protected test split"):
        create_mlx_vlm_smoke_plan(
            TEST_ASSET_ID,
            _config(),
            REGISTRY_PATH,
            MATRIX_PATH,
            ASSET_ROOT,
        )


def test_run_records_validated_test_double_without_aggregating_metrics(tmp_path: Path) -> None:
    record = run_mlx_vlm_smoke(
        DEVELOPMENT_ASSET_ID,
        _config(),
        REGISTRY_PATH,
        MATRIX_PATH,
        ASSET_ROOT,
        inference=_fake_inference,
        executed_at=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )
    output_path = write_mlx_vlm_smoke_record(record, tmp_path)
    summary = summarize_mlx_vlm_smoke_run(record, output_path)

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert record.run_backend == "injected_test_double"
    assert record.validated_response.result_source == "vlm_output"
    assert persisted["request"]["split"] == "development"
    assert persisted["validated_response"]["asset_id"] == DEVELOPMENT_ASSET_ID
    assert summary["performance_claim_supported"] is False
    assert "scores" not in summary


def test_run_rejects_non_json_model_output_and_retains_provenance() -> None:
    raw = "```json\n{}\n```"

    def invalid_inference(_: MlxVlmRunConfig, __: str, ___: Path) -> str:
        return raw

    with pytest.raises(MlxVlmRunnerError, match="exactly one JSON object") as exc_info:
        run_mlx_vlm_smoke(
            DEVELOPMENT_ASSET_ID,
            _config(),
            REGISTRY_PATH,
            MATRIX_PATH,
            ASSET_ROOT,
            inference=invalid_inference,
        )

    error = exc_info.value
    assert error.kind is MlxVlmFailureKind.JSON_PARSE_FAILURE
    assert error.rejected_output is not None
    assert error.rejected_output.preview == raw
    assert error.rejected_output.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_run_rejects_wrong_asset_identity_and_retains_provenance() -> None:
    payload = json.loads(_valid_vlm_output())
    payload["asset_id"] = "fuselage-corrosion-close-diffuse"
    raw = json.dumps(payload)

    def wrong_identity_inference(_: MlxVlmRunConfig, __: str, ___: Path) -> str:
        return raw

    with pytest.raises(MlxVlmRunnerError, match="request binding") as exc_info:
        run_mlx_vlm_smoke(
            DEVELOPMENT_ASSET_ID,
            _config(),
            REGISTRY_PATH,
            MATRIX_PATH,
            ASSET_ROOT,
            inference=wrong_identity_inference,
        )

    error = exc_info.value
    assert error.kind is MlxVlmFailureKind.REQUEST_BINDING_FAILURE
    assert error.rejected_output is not None
    assert error.rejected_output.preview == raw
