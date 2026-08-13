import json

from typer.testing import CliRunner

from aerosynth_eval.cli import app

runner = CliRunner()


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["project"] == "AeroSynth-Eval"


def test_rubric_command_returns_four_dimensions() -> None:
    result = runner.invoke(app, ["rubric"])

    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["dimensions"]) == 4
