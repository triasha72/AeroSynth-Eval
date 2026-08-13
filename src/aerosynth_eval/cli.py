"""Command-line interface for AeroSynth-Eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from aerosynth_eval import __version__
from aerosynth_eval.dataset import load_manifest, summarize_manifest
from aerosynth_eval.rubric import inspection_rubric
from aerosynth_eval.scenario_matrix import load_scenario_matrix, validate_scenario_matrix

app = typer.Typer(
    add_completion=False,
    help="AeroSynth-Eval: evaluation tooling for synthetic aerospace inspection imagery.",
)


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def info() -> None:
    """Print project metadata."""

    _emit(
        {
            "project": "AeroSynth-Eval",
            "version": __version__,
            "status": "scenario_design",
            "scope": "Synthetic aerospace inspection-image evaluation",
        }
    )


@app.command()
def rubric() -> None:
    """Print the fixed v0.1 inspection rubric."""

    _emit(inspection_rubric().model_dump(mode="json"))


@app.command(name="validate-manifest")
def validate_manifest(
    manifest: Annotated[
        Path,
        typer.Argument(
            help="Path to a JSON Lines evaluation manifest.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Validate a versioned evaluation manifest and print its summary."""

    try:
        records = load_manifest(manifest)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="manifest") from error

    summary = summarize_manifest(records)
    _emit({"manifest": str(manifest), **summary.model_dump(mode="json")})


@app.command(name="validate-scenario-matrix")
def validate_scenario_matrix_command(
    matrix: Annotated[
        Path,
        typer.Argument(
            help="Path to a CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Validate the fixed v0.1 synthetic scenario design."""

    try:
        records = load_scenario_matrix(matrix)
        summary = validate_scenario_matrix(records)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="matrix") from error

    _emit({"scenario_matrix": str(matrix), **summary.model_dump(mode="json")})
