"""Optional local MLX-VLM runner for development-only synthetic assets.

The runner validates one explicitly selected development asset, constrains local
MLX-VLM generation to the frozen autograder response schema, validates the
returned response, and records provenance. It never evaluates protected test
assets and does not compute evaluator-quality metrics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosynth_eval.asset_registry import (
    AssetRegistryRecord,
    load_and_validate_materialized_corpus,
    load_asset_registry,
)
from aerosynth_eval.autograder import (
    AutograderRequest,
    build_autograder_request,
    render_runtime_autograder_prompt,
    validate_autograder_response,
)
from aerosynth_eval.contracts import AutograderResponse, AutograderResponseSource

DEFAULT_MLX_VLM_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.0


class MlxVlmFailureKind(StrEnum):
    """Typed failure categories surfaced to higher-level batch orchestration."""

    MODEL_LOAD_FAILURE = "model_load_failure"
    INFERENCE_FAILURE = "inference_failure"
    EMPTY_RESPONSE = "empty_response"
    JSON_PARSE_FAILURE = "json_parse_failure"
    SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"
    REQUEST_BINDING_FAILURE = "request_binding_failure"


class MlxVlmRunnerError(ValueError):
    """Typed local-run failure that preserves the existing ValueError API."""

    kind: MlxVlmFailureKind

    def __init__(self, kind: MlxVlmFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class MlxVlmRunConfig(BaseModel):
    """Explicit local runtime settings recorded with every run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=3, max_length=300)
    max_tokens: int = Field(ge=64, le=2_048)
    temperature: float = Field(ge=0.0, le=1.0)


class MlxVlmSmokePlan(BaseModel):
    """No-inference plan that can be inspected before a local model is loaded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_kind: Literal["local_mlx_vlm_development_smoke"] = "local_mlx_vlm_development_smoke"
    claim_scope: Literal["single_development_inference_only"] = "single_development_inference_only"
    performance_claim_supported: Literal[False] = False
    inference_performed: Literal[False] = False
    request: AutograderRequest
    image_reference: str = Field(min_length=1, max_length=500)
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config: MlxVlmRunConfig


MlxVlmRunBackend = Literal[
    "local_mlx_vlm",
    "shared_local_mlx_vlm",
    "injected_test_double",
]


class MlxVlmSmokeRunRecord(BaseModel):
    """One provenance record from a validated development inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["v0.2"] = "v0.2"
    run_id: str = Field(pattern=r"^mlx-vlm-[a-f0-9]{12}$")
    run_kind: Literal["local_mlx_vlm_development_smoke"] = "local_mlx_vlm_development_smoke"
    run_backend: MlxVlmRunBackend
    claim_scope: Literal["single_development_inference_only"] = "single_development_inference_only"
    performance_claim_supported: Literal[False] = False
    inference_performed: Literal[True] = True
    executed_at: datetime
    runner: Literal["mlx-vlm"] = "mlx-vlm"
    runner_version: str = Field(min_length=1, max_length=100)
    config: MlxVlmRunConfig
    request: AutograderRequest
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt: str = Field(min_length=1, max_length=50_000)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_model_output: str = Field(min_length=1, max_length=30_000)
    validated_response: AutograderResponse


MlxVlmInference = Callable[[MlxVlmRunConfig, str, Path], str]
MlxVlmSessionFactory = Callable[[MlxVlmRunConfig], MlxVlmInference]
MlxLogitsProcessor = Callable[[Any, Any], Any]
MlxStructuredOutputBuilder = Callable[
    [Any, dict[str, Any]],
    MlxLogitsProcessor,
]


@dataclass(frozen=True)
class _PreparedMlxVlmSmoke:
    """Internal execution inputs that are not printed by the dry-run command."""

    plan: MlxVlmSmokePlan
    prompt: str
    image_path: Path


def _safe_asset_path(asset_root: Path, image_reference: str) -> Path:
    """Resolve a registry reference below an explicit root without path traversal."""

    reference = Path(image_reference)
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError(f"Unsafe image reference '{image_reference}'.")
    return asset_root / reference


def _find_asset(asset_id: str, registry_path: Path) -> AssetRegistryRecord:
    """Find one registry record after the public request builder validated its identity."""

    asset = next(
        (record for record in load_asset_registry(registry_path) if record.asset_id == asset_id),
        None,
    )
    if asset is None:
        raise ValueError(f"Unknown asset_id '{asset_id}'.")
    if asset.image_sha256 is None:
        raise ValueError(f"Generated asset '{asset_id}' is missing an image SHA-256 digest.")
    return asset


def _prepare_mlx_vlm_smoke(
    asset_id: str,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
) -> _PreparedMlxVlmSmoke:
    """Validate corpus integrity and prepare one local development-only invocation."""

    request = build_autograder_request(asset_id, registry_path, scenario_matrix_path)
    load_and_validate_materialized_corpus(registry_path, scenario_matrix_path, asset_root)
    asset = _find_asset(asset_id, registry_path)
    image_path = _safe_asset_path(asset_root, request.image_reference)
    try:
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"Could not read generated development image '{image_path}'.") from error
    if image_sha256 != asset.image_sha256:
        raise ValueError(
            f"Generated image '{image_path}' does not match its recorded SHA-256 digest."
        )

    prompt = render_runtime_autograder_prompt(request)
    plan = MlxVlmSmokePlan(
        request=request,
        image_reference=request.image_reference,
        image_sha256=image_sha256,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        config=config,
    )
    return _PreparedMlxVlmSmoke(plan=plan, prompt=prompt, image_path=image_path)


def create_mlx_vlm_smoke_plan(
    asset_id: str,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
) -> MlxVlmSmokePlan:
    """Return a fully validated plan without importing or invoking MLX-VLM."""

    return _prepare_mlx_vlm_smoke(
        asset_id,
        config,
        registry_path,
        scenario_matrix_path,
        asset_root,
    ).plan


def _require_apple_silicon() -> None:
    """Fail before model loading on machines unsupported by this local runner."""

    machine = platform.machine().lower()
    if platform.system() != "Darwin" or machine not in {"arm64", "arm64e"}:
        raise ValueError(
            "The local MLX-VLM runner requires macOS on Apple Silicon "
            f"(detected {platform.system()} {platform.machine()})."
        )


def _extract_mlx_vlm_text(output: object) -> str:
    """Extract generated text from supported MLX-VLM return shapes."""

    text: object

    if isinstance(output, str):
        text = output
    else:
        text = getattr(output, "text", None)

    if not isinstance(text, str) or not text.strip():
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.EMPTY_RESPONSE,
            "MLX-VLM returned no non-empty text output.",
        )

    return text


def _build_autograder_logits_processor(
    processor: Any,
    builder: MlxStructuredOutputBuilder,
) -> MlxLogitsProcessor:
    """Build constrained decoding from the frozen autograder response schema."""

    tokenizer = getattr(processor, "tokenizer", processor)
    schema = AutograderResponse.model_json_schema()
    return builder(tokenizer, schema)


def create_mlx_vlm_session(config: MlxVlmRunConfig) -> MlxVlmInference:
    """Load MLX-VLM once and return an inference callable reusable across batch cases."""

    try:
        _require_apple_silicon()
    except ValueError as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            str(error),
        ) from error

    try:
        mlx_vlm = import_module("mlx_vlm")
        prompt_utils = import_module("mlx_vlm.prompt_utils")
        structured = import_module("mlx_vlm.structured")
        utils = import_module("mlx_vlm.utils")
    except ModuleNotFoundError as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            'MLX-VLM is not installed. Run: python -m pip install -e ".[dev,mlx]"',
        ) from error

    load = cast(Callable[[str], tuple[Any, Any]], mlx_vlm.load)
    generate = cast(Callable[..., object], mlx_vlm.generate)
    apply_chat_template = cast(Callable[..., str], prompt_utils.apply_chat_template)
    build_json_schema_logits_processor = cast(
        MlxStructuredOutputBuilder,
        structured.build_json_schema_logits_processor,
    )
    load_config = cast(Callable[[str], Any], utils.load_config)

    try:
        model, processor = load(config.model_id)
        model_config = load_config(config.model_id)
    except Exception as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            f"MLX-VLM model/session initialization failed: {error}",
        ) from error

    def session_inference(
        runtime_config: MlxVlmRunConfig,
        prompt: str,
        image_path: Path,
    ) -> str:
        if runtime_config != config:
            raise ValueError("Shared MLX-VLM session configuration changed during a batch run.")

        try:
            # Build a fresh constrained decoder for every response so parser state is
            # never shared across independent development cases.
            json_logits_processor = _build_autograder_logits_processor(
                processor,
                build_json_schema_logits_processor,
            )
            formatted_prompt = apply_chat_template(
                processor,
                model_config,
                prompt,
                num_images=1,
            )
            output = generate(
                model,
                processor,
                formatted_prompt,
                image=[str(image_path)],
                verbose=False,
                max_tokens=runtime_config.max_tokens,
                temperature=runtime_config.temperature,
                logits_processors=[json_logits_processor],
            )
        except Exception as error:
            raise MlxVlmRunnerError(
                MlxVlmFailureKind.INFERENCE_FAILURE,
                f"MLX-VLM inference failed: {error}",
            ) from error

        return _extract_mlx_vlm_text(output)

    return session_inference


def _run_with_local_mlx_vlm(config: MlxVlmRunConfig, prompt: str, image_path: Path) -> str:
    """Run one standalone local inference using a one-call session."""

    inference = create_mlx_vlm_session(config)
    return inference(config, prompt, image_path)


def _parse_vlm_output(raw_model_output: str, request: AutograderRequest) -> AutograderResponse:
    """Require one exact JSON object before accepting a model response."""

    try:
        payload: object = json.loads(raw_model_output)
    except json.JSONDecodeError as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.JSON_PARSE_FAILURE,
            "MLX-VLM output must be exactly one JSON object with no Markdown fence or prose.",
        ) from error

    if not isinstance(payload, dict):
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.JSON_PARSE_FAILURE,
            "MLX-VLM output must be a JSON object.",
        )

    try:
        response = AutograderResponse.model_validate(payload)
    except ValidationError as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.SCHEMA_VALIDATION_FAILURE,
            f"MLX-VLM output violates the autograder response contract: {error}",
        ) from error

    if response.result_source is not AutograderResponseSource.VLM_OUTPUT:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.SCHEMA_VALIDATION_FAILURE,
            "MLX-VLM output must declare result_source 'vlm_output'.",
        )

    try:
        return validate_autograder_response(response, request)
    except ValueError as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.REQUEST_BINDING_FAILURE,
            f"MLX-VLM output failed request binding: {error}",
        ) from error


def _normalized_execution_time(executed_at: datetime | None) -> datetime:
    """Return a UTC timestamp and reject an ambiguous caller-supplied test value."""

    timestamp = executed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("executed_at must include a UTC offset.")
    return timestamp.astimezone(UTC).replace(microsecond=0)


def _installed_mlx_vlm_version() -> str:
    """Capture the installed MLX-VLM package version after a real local invocation."""

    try:
        return metadata.version("mlx-vlm")
    except metadata.PackageNotFoundError:
        return "unknown"


def run_mlx_vlm_smoke(
    asset_id: str,
    config: MlxVlmRunConfig,
    registry_path: Path,
    scenario_matrix_path: Path,
    asset_root: Path,
    *,
    inference: MlxVlmInference | None = None,
    inference_backend: MlxVlmRunBackend | None = None,
    executed_at: datetime | None = None,
) -> MlxVlmSmokeRunRecord:
    """Run and validate one development-only VLM response without aggregation."""

    prepared = _prepare_mlx_vlm_smoke(
        asset_id,
        config,
        registry_path,
        scenario_matrix_path,
        asset_root,
    )

    inference_function: MlxVlmInference

    if inference is None:
        if inference_backend is not None:
            raise ValueError("inference_backend requires an injected inference callable.")
        inference_function = _run_with_local_mlx_vlm
        run_backend: MlxVlmRunBackend = "local_mlx_vlm"
    else:
        inference_function = inference
        run_backend = inference_backend or "injected_test_double"
        if run_backend == "local_mlx_vlm":
            raise ValueError(
                "Injected inference cannot declare the standalone local_mlx_vlm backend."
            )

    raw_model_output = inference_function(config, prepared.prompt, prepared.image_path)
    response = _parse_vlm_output(raw_model_output, prepared.plan.request)

    return MlxVlmSmokeRunRecord(
        run_id=f"mlx-vlm-{uuid.uuid4().hex[:12]}",
        run_backend=run_backend,
        executed_at=_normalized_execution_time(executed_at),
        runner_version=(
            "test-double" if run_backend == "injected_test_double" else _installed_mlx_vlm_version()
        ),
        config=config,
        request=prepared.plan.request,
        image_sha256=prepared.plan.image_sha256,
        prompt=prepared.prompt,
        prompt_sha256=prepared.plan.prompt_sha256,
        raw_model_output=raw_model_output,
        validated_response=response,
    )


def write_mlx_vlm_smoke_record(
    record: MlxVlmSmokeRunRecord,
    output_directory: Path,
) -> Path:
    """Persist a full provenance record without overwriting a prior local run."""

    output_path = output_directory / f"{record.run_id}.json"
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing MLX-VLM run record '{output_path}'.")
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(f"Could not write MLX-VLM run record to '{output_path}'.") from error
    return output_path


def summarize_mlx_vlm_smoke_run(
    record: MlxVlmSmokeRunRecord,
    output_path: Path,
) -> dict[str, object]:
    """Print run provenance without exposing rationales or aggregating performance."""

    return {
        "run_record": str(output_path),
        "run_id": record.run_id,
        "run_backend": record.run_backend,
        "claim_scope": record.claim_scope,
        "performance_claim_supported": record.performance_claim_supported,
        "inference_performed": record.inference_performed,
        "model_id": record.config.model_id,
        "mlx_vlm_version": record.runner_version,
        "asset_id": record.request.asset_id,
        "scenario_id": record.request.scenario_id,
        "split": record.request.split,
        "rubric_version": record.request.rubric_version,
        "result_source": record.validated_response.result_source,
        "decision": record.validated_response.decision,
        "scored_dimensions": [score.dimension for score in record.validated_response.scores],
    }
