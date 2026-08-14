import json
from pathlib import Path

from typer.testing import CliRunner

from aerosynth_eval.cli import app

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "examples" / "v0_1_manifest.jsonl"
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"
ANNOTATION_QUEUE_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "v0_1_development_annotation_queue.csv"
)
SYNTHETIC_DEMO_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_demo"
SYNTHETIC_DEMO_RATER_A_PATH = SYNTHETIC_DEMO_FIXTURE_ROOT / "synthetic_demo_rater_a.csv"
SYNTHETIC_DEMO_RATER_B_PATH = SYNTHETIC_DEMO_FIXTURE_ROOT / "synthetic_demo_rater_b.csv"
AUTOGRADER_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "autograder"
AUTOGRADER_RESPONSE_PATH = AUTOGRADER_FIXTURE_ROOT / "synthetic_valid_response.json"
DEVELOPMENT_AUTOGRADER_ASSET_ID = "asset-fuselage-corrosion-close-diffuse"
TEST_AUTOGRADER_ASSET_ID = "asset-fuselage-clean-close-diffuse"


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["project"] == "AeroSynth-Eval"


def test_rubric_command_returns_four_dimensions() -> None:
    result = runner.invoke(app, ["rubric"])

    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["dimensions"]) == 4


def test_preview_autograder_prompt_command_returns_development_only_request() -> None:
    result = runner.invoke(app, ["preview-autograder-prompt", DEVELOPMENT_AUTOGRADER_ASSET_ID])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["request"]["asset_id"] == DEVELOPMENT_AUTOGRADER_ASSET_ID
    assert payload["request"]["split"] == "development"
    assert "performs no model inference" in payload["prompt"]


def test_preview_autograder_prompt_command_rejects_protected_test_asset() -> None:
    result = runner.invoke(app, ["preview-autograder-prompt", TEST_AUTOGRADER_ASSET_ID])

    assert result.exit_code != 0
    assert "protected test split" in result.output


def test_validate_autograder_response_command_returns_schema_only_summary() -> None:
    result = runner.invoke(app, ["validate-autograder-response", str(AUTOGRADER_RESPONSE_PATH)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["contract_status"] == "schema_validated_only"
    assert payload["result_source"] == "synthetic_contract_fixture"
    assert payload["inference_performed_by_this_command"] is False


def test_validate_manifest_command_returns_summary() -> None:
    result = runner.invoke(app, ["validate-manifest", str(MANIFEST_PATH)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 6
    assert payload["splits"]["development"] == 4
    assert payload["splits"]["test"] == 2


def test_validate_scenario_matrix_command_returns_summary() -> None:
    result = runner.invoke(app, ["validate-scenario-matrix", str(MATRIX_PATH)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 48
    assert payload["splits"]["development"] == 36
    assert payload["splits"]["test"] == 12


def test_validate_asset_registry_command_returns_summary() -> None:
    result = runner.invoke(
        app,
        [
            "validate-asset-registry",
            str(REGISTRY_PATH),
            "--scenario-matrix",
            str(MATRIX_PATH),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 48
    assert payload["lifecycle_statuses"]["generated"] == 48
    assert payload["records_with_generation_evidence"] == 48


def test_validate_annotation_queue_command_returns_summary() -> None:
    result = runner.invoke(app, ["validate-annotation-queue", str(ANNOTATION_QUEUE_PATH)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 12
    assert payload["split"] == "development"
    assert payload["capture_profiles"] == {
        "close_diffuse": 3,
        "close_glare": 3,
        "oblique_blur": 3,
        "oblique_directional": 3,
    }


def test_summarize_synthetic_agreement_command_returns_fixture_summary() -> None:
    result = runner.invoke(
        app,
        [
            "summarize-synthetic-agreement",
            str(SYNTHETIC_DEMO_RATER_A_PATH),
            str(SYNTHETIC_DEMO_RATER_B_PATH),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data_classification"] == "synthetic_demo_fixture"
    assert payload["human_agreement_claim_supported"] is False
    assert payload["record_count"] == 12
    assert payload["decision_exact_agreement"]["count"] == 10
    assert len(payload["disagreement_candidates"]) == 2
