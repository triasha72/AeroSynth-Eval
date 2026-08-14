from pathlib import Path

import pytest

from aerosynth_eval.autograder import (
    build_autograder_request,
    load_autograder_response,
    render_autograder_prompt,
    summarize_validated_autograder_response,
    validate_autograder_response,
)
from aerosynth_eval.contracts import (
    AutograderResponseSource,
    DatasetSplit,
    EvaluationDecision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "autograder"
VALID_RESPONSE_PATH = FIXTURE_ROOT / "synthetic_valid_response.json"
INVALID_RESPONSE_PATH = FIXTURE_ROOT / "invalid_response_missing_rationale.json"
DEVELOPMENT_ASSET_ID = "asset-fuselage-corrosion-close-diffuse"
TEST_ASSET_ID = "asset-fuselage-clean-close-diffuse"


def test_build_autograder_request_uses_frozen_development_metadata() -> None:
    request = build_autograder_request(DEVELOPMENT_ASSET_ID, REGISTRY_PATH, MATRIX_PATH)

    assert request.split is DatasetSplit.DEVELOPMENT
    assert request.scenario_id == "fuselage-corrosion-close-diffuse"
    assert request.rubric_version == "v0.1"


def test_build_autograder_request_rejects_protected_test_asset() -> None:
    with pytest.raises(ValueError, match="protected test split"):
        build_autograder_request(TEST_ASSET_ID, REGISTRY_PATH, MATRIX_PATH)


def test_render_autograder_prompt_contains_request_and_output_schema() -> None:
    request = build_autograder_request(DEVELOPMENT_ASSET_ID, REGISTRY_PATH, MATRIX_PATH)
    prompt = render_autograder_prompt(request)

    assert DEVELOPMENT_ASSET_ID in prompt
    assert '"output_schema"' in prompt
    assert "performs no model inference" in prompt


def test_valid_synthetic_response_is_schema_validated_only() -> None:
    request = build_autograder_request(DEVELOPMENT_ASSET_ID, REGISTRY_PATH, MATRIX_PATH)
    response = load_autograder_response(VALID_RESPONSE_PATH)

    summary = summarize_validated_autograder_response(response, request)

    assert response.result_source is AutograderResponseSource.SYNTHETIC_CONTRACT_FIXTURE
    assert response.decision is EvaluationDecision.UNCERTAIN
    assert summary["contract_status"] == "schema_validated_only"
    assert summary["inference_performed_by_this_command"] is False
    assert len(summary["scored_dimensions"]) == 4


def test_invalid_response_rejects_missing_rationale() -> None:
    with pytest.raises(ValueError, match="invalid autograder response"):
        load_autograder_response(INVALID_RESPONSE_PATH)


def test_response_must_match_the_grounded_request() -> None:
    request = build_autograder_request(DEVELOPMENT_ASSET_ID, REGISTRY_PATH, MATRIX_PATH)
    response = load_autograder_response(VALID_RESPONSE_PATH)
    mismatched_response = response.model_copy(
        update={"scenario_id": "wing-corrosion-close-diffuse"}
    )

    with pytest.raises(ValueError, match="scenario_id"):
        validate_autograder_response(mismatched_response, request)
