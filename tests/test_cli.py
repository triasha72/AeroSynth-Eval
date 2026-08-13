import json
from pathlib import Path

from typer.testing import CliRunner

from aerosynth_eval.cli import app

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "examples" / "v0_1_manifest.jsonl"
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["project"] == "AeroSynth-Eval"


def test_rubric_command_returns_four_dimensions() -> None:
    result = runner.invoke(app, ["rubric"])

    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["dimensions"]) == 4


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
