"""Command-line interface for AeroSynth-Eval."""

from __future__ import annotations

import json
from typing import Any

import typer

from aerosynth_eval import __version__
from aerosynth_eval.rubric import inspection_rubric

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
            "status": "foundation",
            "scope": "Synthetic aerospace inspection-image evaluation",
        }
    )


@app.command()
def rubric() -> None:
    """Print the fixed v0.1 inspection rubric."""

    _emit(inspection_rubric().model_dump(mode="json"))
