"""VLM-ready prompt and response contracts for development-only assets."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosynth_eval.asset_registry import (
    AssetRegistryRecord,
    load_asset_registry,
    validate_asset_registry,
)
from aerosynth_eval.contracts import (
    AircraftRegion,
    AssetLifecycleStatus,
    AutograderResponse,
    DatasetSplit,
    SurfaceCondition,
)
from aerosynth_eval.rubric import inspection_rubric
from aerosynth_eval.scenario_matrix import ScenarioMatrixRecord, load_scenario_matrix


class AutograderRequest(BaseModel):
    """Grounded context supplied to a future VLM evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: DatasetSplit
    image_reference: str = Field(min_length=1, max_length=500)
    component: AircraftRegion
    expected_condition: SurfaceCondition
    capture_profile: str = Field(min_length=3, max_length=64)
    viewpoint: str = Field(min_length=3, max_length=120)
    lighting: str = Field(min_length=3, max_length=120)
    quality_challenge: str = Field(min_length=2, max_length=64)
    rubric_version: str = Field(min_length=1, max_length=32)


def _resolve_development_asset(
    asset_id: str,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> tuple[AssetRegistryRecord, ScenarioMatrixRecord]:
    """Resolve one generated development asset against frozen provenance."""

    registry = load_asset_registry(registry_path)
    scenarios = load_scenario_matrix(scenario_matrix_path)
    validate_asset_registry(registry, scenarios)

    asset = next((record for record in registry if record.asset_id == asset_id), None)
    if asset is None:
        raise ValueError(f"Unknown asset_id '{asset_id}'.")
    if asset.split is not DatasetSplit.DEVELOPMENT:
        raise ValueError(
            f"Asset '{asset_id}' belongs to the protected test split and cannot be used here."
        )
    if asset.lifecycle_status is not AssetLifecycleStatus.GENERATED:
        raise ValueError(f"Asset '{asset_id}' must be generated before prompt preview.")

    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    scenario = scenarios_by_id[asset.scenario_id]
    return asset, scenario


def build_autograder_request(
    asset_id: str,
    registry_path: Path,
    scenario_matrix_path: Path,
) -> AutograderRequest:
    """Build a development-only request from the frozen asset and scenario metadata."""

    asset, scenario = _resolve_development_asset(asset_id, registry_path, scenario_matrix_path)
    rubric = inspection_rubric()
    return AutograderRequest(
        asset_id=asset.asset_id,
        scenario_id=scenario.scenario_id,
        split=asset.split,
        image_reference=asset.image_reference,
        component=scenario.component,
        expected_condition=scenario.condition,
        capture_profile=scenario.capture_profile,
        viewpoint=scenario.viewpoint,
        lighting=scenario.lighting,
        quality_challenge=scenario.quality_challenge,
        rubric_version=rubric.version,
    )


def render_autograder_prompt(request: AutograderRequest) -> str:
    """Render a deterministic, schema-grounded prompt without invoking a model."""

    payload = {
        "task": (
            "Evaluate one synthetic research image against the supplied rubric. "
            "Return only one JSON object that satisfies output_schema."
        ),
        "scope_boundary": (
            "This is a non-operational research-data workflow. Do not provide maintenance, "
            "airworthiness, certification, or defect-diagnosis advice."
        ),
        "image_handling": (
            "A future runner must attach the image at image_reference. This command previews "
            "the prompt only and performs no model inference."
        ),
        "request": request.model_dump(mode="json"),
        "rubric": inspection_rubric().model_dump(mode="json"),
        "output_schema": AutograderResponse.model_json_schema(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def load_autograder_response(path: Path) -> AutograderResponse:
    """Load one strict JSON response and retain useful source-path errors."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Could not read autograder response at {path}.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}.") from error

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: autograder response must be a JSON object.")
    try:
        return AutograderResponse.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"{path}: invalid autograder response: {error}") from error


def validate_autograder_response(
    response: AutograderResponse,
    request: AutograderRequest,
) -> AutograderResponse:
    """Ensure a response is bound to exactly the development request it claims to grade."""

    for field_name, response_value, request_value in (
        ("asset_id", response.asset_id, request.asset_id),
        ("scenario_id", response.scenario_id, request.scenario_id),
        ("rubric_version", response.rubric_version, request.rubric_version),
    ):
        if response_value != request_value:
            raise ValueError(
                f"Autograder response {field_name} '{response_value}' does not match "
                f"development request value '{request_value}'."
            )
    return response


def summarize_validated_autograder_response(
    response: AutograderResponse,
    request: AutograderRequest,
) -> dict[str, object]:
    """Emit contract-validation metadata without reporting rationale text or metrics."""

    validate_autograder_response(response, request)
    return {
        "contract_status": "schema_validated_only",
        "claim_scope": "response_schema_validation_only",
        "inference_performed_by_this_command": False,
        "asset_id": request.asset_id,
        "scenario_id": request.scenario_id,
        "split": request.split,
        "rubric_version": request.rubric_version,
        "result_source": response.result_source,
        "decision": response.decision,
        "scored_dimensions": [score.dimension for score in response.scores],
    }
