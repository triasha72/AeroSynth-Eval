"""Command-line interface for AeroSynth-Eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from aerosynth_eval import __version__
from aerosynth_eval.agreement import summarize_synthetic_demo_agreement
from aerosynth_eval.annotations import (
    load_and_validate_annotation_queue,
    load_and_validate_rater_annotations,
    load_rater_annotations,
)
from aerosynth_eval.asset_registry import (
    load_and_validate_asset_registry,
    load_and_validate_materialized_corpus,
)
from aerosynth_eval.autograder import (
    build_autograder_request,
    load_autograder_response,
    render_autograder_prompt,
    summarize_validated_autograder_response,
)
from aerosynth_eval.dataset import load_manifest, summarize_manifest
from aerosynth_eval.human_annotation_ingestion import (
    ingest_human_annotation_study,
    prepare_human_rater_template,
    summarize_human_annotation_ingestion,
    write_human_annotation_ingestion_manifest,
)
from aerosynth_eval.mlx_vlm_runner import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MLX_VLM_MODEL,
    DEFAULT_TEMPERATURE,
    MlxVlmRunConfig,
    create_mlx_vlm_smoke_plan,
    run_mlx_vlm_smoke,
    summarize_mlx_vlm_smoke_run,
    write_mlx_vlm_smoke_record,
)
from aerosynth_eval.procedural_corpus import materialize_procedural_corpus
from aerosynth_eval.rubric import inspection_rubric
from aerosynth_eval.scenario_matrix import load_scenario_matrix, validate_scenario_matrix
from aerosynth_eval.vlm_batch_runner import (
    DEFAULT_BATCH_MAX_RETRIES,
    VlmBatchRetryPolicy,
    create_development_vlm_batch_plan,
    run_development_vlm_batch,
    summarize_development_vlm_batch,
    write_development_vlm_batch_record,
)

DEFAULT_SCENARIO_MATRIX = Path("data/design/v0_1_scenario_matrix.csv")
DEFAULT_ASSET_REGISTRY = Path("data/registry/v0_1_asset_registry.jsonl")
DEFAULT_ASSET_ROOT = Path("data")
DEFAULT_ANNOTATION_QUEUE = Path("data/annotations/v0_1_development_annotation_queue.csv")
DEFAULT_MLX_VLM_OUTPUT_DIRECTORY = Path("outputs/mlx_vlm")
DEFAULT_VLM_BATCH_OUTPUT_DIRECTORY = Path("outputs/vlm_batches")
DEFAULT_HUMAN_ANNOTATION_OUTPUT_DIRECTORY = Path("outputs/human_annotations")

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
            "status": "multi_judge",
            "scope": "Synthetic aerospace inspection-image evaluation",
        }
    )


@app.command()
def rubric() -> None:
    """Print the fixed v0.1 inspection rubric."""

    _emit(inspection_rubric().model_dump(mode="json"))


@app.command(name="preview-autograder-prompt")
def preview_autograder_prompt_command(
    asset_id: Annotated[
        str,
        typer.Argument(
            help="Development-split asset ID to ground the future VLM prompt.",
        ),
    ],
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Preview a grounded prompt for a generated development asset; never run a model."""

    try:
        request = build_autograder_request(asset_id, registry, scenario_matrix)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="asset_id") from error

    _emit(
        {
            "request": request.model_dump(mode="json"),
            "prompt": render_autograder_prompt(request),
        }
    )


@app.command(name="validate-autograder-response")
def validate_autograder_response_command(
    response_path: Annotated[
        Path,
        typer.Argument(
            help="Path to one strict autograder-response JSON document.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Validate a response against one development request without evaluating a VLM."""

    try:
        response = load_autograder_response(response_path)
        request = build_autograder_request(response.asset_id, registry, scenario_matrix)
        summary = summarize_validated_autograder_response(response, request)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="response_path") from error

    _emit({"autograder_response": str(response_path), **summary})


@app.command(name="run-mlx-vlm-smoke")
def run_mlx_vlm_smoke_command(
    asset_id: Annotated[
        str,
        typer.Argument(
            help="Generated development asset ID for one local VLM smoke inference.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="MLX or Hugging Face model identifier to load locally."),
    ] = DEFAULT_MLX_VLM_MODEL,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Maximum output tokens for this one local run."),
    ] = DEFAULT_MAX_TOKENS,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Sampling temperature for this one local run."),
    ] = DEFAULT_TEMPERATURE,
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-directory",
            help="Ignored directory where a successful local provenance record is written.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = DEFAULT_MLX_VLM_OUTPUT_DIRECTORY,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the development request and image without importing or running MLX-VLM.",
        ),
    ] = False,
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
    asset_root: Annotated[
        Path,
        typer.Option(
            "--asset-root",
            help="Directory that contains the registry's assets/v0_1 paths.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = DEFAULT_ASSET_ROOT,
) -> None:
    """Run one local VLM smoke inference or inspect its development-only plan."""

    try:
        config = MlxVlmRunConfig(
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if dry_run:
            plan = create_mlx_vlm_smoke_plan(
                asset_id,
                config,
                registry,
                scenario_matrix,
                asset_root,
            )
            _emit({"run_plan": plan.model_dump(mode="json")})
            return

        record = run_mlx_vlm_smoke(
            asset_id,
            config,
            registry,
            scenario_matrix,
            asset_root,
        )
        output_path = write_mlx_vlm_smoke_record(record, output_directory)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="asset_id") from error

    _emit(summarize_mlx_vlm_smoke_run(record, output_path))


@app.command(name="run-vlm-batch")
def run_vlm_batch_command(
    queue: Annotated[
        Path,
        typer.Option(
            "--queue",
            help="Path to the fixed development-only evaluation queue.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ANNOTATION_QUEUE,
    model: Annotated[
        str,
        typer.Option("--model", help="MLX or Hugging Face model identifier to load locally."),
    ] = DEFAULT_MLX_VLM_MODEL,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Maximum output tokens for each development case."),
    ] = DEFAULT_MAX_TOKENS,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Sampling temperature for each development case."),
    ] = DEFAULT_TEMPERATURE,
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            help="Bounded retries for transient inference failures only.",
            min=0,
            max=3,
        ),
    ] = DEFAULT_BATCH_MAX_RETRIES,
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-directory",
            help="Ignored directory where the batch provenance record is written.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = DEFAULT_VLM_BATCH_OUTPUT_DIRECTORY,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the complete development batch without running MLX-VLM.",
        ),
    ] = False,
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
    asset_root: Annotated[
        Path,
        typer.Option(
            "--asset-root",
            help="Directory containing the registry's generated image paths.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = DEFAULT_ASSET_ROOT,
) -> None:
    """Run or inspect the fixed development-only VLM evaluation batch."""

    try:
        config = MlxVlmRunConfig(
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        retry_policy = VlmBatchRetryPolicy(max_retries=max_retries)

        if dry_run:
            plan = create_development_vlm_batch_plan(
                queue,
                config,
                registry,
                scenario_matrix,
                asset_root,
                retry_policy=retry_policy,
            )
            _emit({"batch_plan": plan.model_dump(mode="json")})
            return

        record = run_development_vlm_batch(
            queue,
            config,
            registry,
            scenario_matrix,
            asset_root,
            retry_policy=retry_policy,
        )
        output_path = write_development_vlm_batch_record(
            record,
            output_directory,
        )
        summary = summarize_development_vlm_batch(record)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="queue") from error

    _emit(
        {
            "batch_record": str(output_path),
            **summary.model_dump(mode="json"),
        }
    )


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


@app.command(name="validate-asset-registry")
def validate_asset_registry_command(
    registry: Annotated[
        Path,
        typer.Argument(
            help="Path to a JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Validate synthetic-asset provenance against the frozen scenario design."""

    try:
        summary = load_and_validate_asset_registry(registry, scenario_matrix)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="registry") from error

    _emit(
        {
            "asset_registry": str(registry),
            "scenario_matrix": str(scenario_matrix),
            **summary.model_dump(mode="json"),
        }
    )


@app.command(name="validate-annotation-queue")
def validate_annotation_queue_command(
    queue: Annotated[
        Path,
        typer.Argument(
            help="Path to the development-only human-annotation queue CSV.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Validate a balanced development-only queue before human rating begins."""

    try:
        summary = load_and_validate_annotation_queue(queue, registry, scenario_matrix)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="queue") from error
    _emit(
        {
            "annotation_queue": str(queue),
            "asset_registry": str(registry),
            "scenario_matrix": str(scenario_matrix),
            **summary.model_dump(mode="json"),
        }
    )


@app.command(name="validate-rater-annotations")
def validate_rater_annotations_command(
    annotations: Annotated[
        Path,
        typer.Argument(
            help="Path to a complete pseudonymous rater-annotation CSV submission.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    queue: Annotated[
        Path,
        typer.Option(
            "--queue",
            help="Path to the approved development-only annotation queue.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ANNOTATION_QUEUE,
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Validate one or more complete human-rater submissions without scoring agreement."""

    try:
        summary = load_and_validate_rater_annotations(
            annotations,
            queue,
            registry,
            scenario_matrix,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="annotations") from error
    _emit(
        {
            "rater_annotations": str(annotations),
            "annotation_queue": str(queue),
            **summary.model_dump(mode="json"),
        }
    )


@app.command(name="prepare-human-rater-template")
def prepare_human_rater_template_command(
    rater_id: Annotated[
        str,
        typer.Argument(
            help="Pseudonymous rater ID; do not use a person's direct identifier.",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Private CSV path to create; existing files are never overwritten.",
        ),
    ],
    queue: Annotated[
        Path,
        typer.Option(
            "--queue",
            help="Path to the approved development-only annotation queue.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ANNOTATION_QUEUE,
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Create one blank private rater CSV bound to the development queue."""

    try:
        created_path = prepare_human_rater_template(
            rater_id,
            output_path,
            queue,
            registry,
            scenario_matrix,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="rater_id") from error

    _emit(
        {
            "template": str(created_path),
            "rater_id": rater_id,
            "annotation_queue": str(queue),
            "split": "development",
            "ratings_populated": False,
            "agreement_computed": False,
        }
    )


@app.command(name="ingest-human-annotations")
def ingest_human_annotations_command(
    annotations_a: Annotated[
        Path,
        typer.Argument(
            help="Private complete CSV submission from the first pseudonymous rater.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    annotations_b: Annotated[
        Path,
        typer.Argument(
            help="Private complete CSV submission from the second pseudonymous rater.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    queue: Annotated[
        Path,
        typer.Option(
            "--queue",
            help="Path to the approved development-only annotation queue.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ANNOTATION_QUEUE,
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-directory",
            help="Ignored directory for the non-label ingestion manifest.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = DEFAULT_HUMAN_ANNOTATION_OUTPUT_DIRECTORY,
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Validate and fingerprint two complete private human-rater submissions."""

    try:
        manifest = ingest_human_annotation_study(
            annotations_a,
            annotations_b,
            queue,
            registry,
            scenario_matrix,
        )
        manifest_path = write_human_annotation_ingestion_manifest(
            manifest,
            output_directory,
        )
        summary = summarize_human_annotation_ingestion(manifest)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="annotations_a") from error

    _emit(
        {
            "ingestion_manifest": str(manifest_path),
            **summary.model_dump(mode="json"),
        }
    )


@app.command(name="summarize-synthetic-agreement")
def summarize_synthetic_agreement_command(
    annotations_a: Annotated[
        Path,
        typer.Argument(
            help="Path to one explicitly synthetic demo rater-annotation CSV fixture.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    annotations_b: Annotated[
        Path,
        typer.Argument(
            help="Path to a second explicitly synthetic demo rater-annotation CSV fixture.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    queue: Annotated[
        Path,
        typer.Option(
            "--queue",
            help="Path to the approved development-only annotation queue.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ANNOTATION_QUEUE,
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
) -> None:
    """Summarize only clearly marked synthetic fixtures, never human submissions."""

    try:
        load_and_validate_rater_annotations(
            annotations_a,
            queue,
            registry,
            scenario_matrix,
        )
        load_and_validate_rater_annotations(
            annotations_b,
            queue,
            registry,
            scenario_matrix,
        )
        summary = summarize_synthetic_demo_agreement(
            load_rater_annotations(annotations_a),
            load_rater_annotations(annotations_b),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="annotations") from error

    _emit(
        {
            "annotations_a": str(annotations_a),
            "annotations_b": str(annotations_b),
            "annotation_queue": str(queue),
            **summary.model_dump(mode="json"),
        }
    )


@app.command(name="materialize-procedural-corpus")
def materialize_procedural_corpus_command(
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines synthetic-image asset registry to update.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
    asset_root: Annotated[
        Path,
        typer.Option(
            "--asset-root",
            help="Directory that contains the registry's assets/v0_1 paths.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = DEFAULT_ASSET_ROOT,
) -> None:
    """Render all planned v0.1 assets with the deterministic procedural baseline."""

    try:
        summary = materialize_procedural_corpus(registry, scenario_matrix, asset_root)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="registry") from error
    _emit(summary.model_dump(mode="json"))


@app.command(name="validate-materialized-corpus")
def validate_materialized_corpus_command(
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the JSON Lines generated-asset registry.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_ASSET_REGISTRY,
    scenario_matrix: Annotated[
        Path,
        typer.Option(
            "--scenario-matrix",
            help="Path to the frozen CSV scenario matrix.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_SCENARIO_MATRIX,
    asset_root: Annotated[
        Path,
        typer.Option(
            "--asset-root",
            help="Directory that contains the registry's assets/v0_1 paths.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = DEFAULT_ASSET_ROOT,
) -> None:
    """Verify all bundled generated assets against their frozen provenance."""

    try:
        summary = load_and_validate_materialized_corpus(
            registry,
            scenario_matrix,
            asset_root,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="registry") from error
    _emit(
        {
            "asset_registry": str(registry),
            "scenario_matrix": str(scenario_matrix),
            "asset_root": str(asset_root),
            **summary.model_dump(mode="json"),
        }
    )
