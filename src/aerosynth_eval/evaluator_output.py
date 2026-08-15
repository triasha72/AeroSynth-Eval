"""Request-bound structured-output helpers and rejected-output provenance."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.contracts import AutograderResponse, AutograderResponseSource

REJECTED_OUTPUT_PREVIEW_MAX_CHARS = 4_000


class RejectedOutputProvenance(BaseModel):
    """Bounded diagnostic provenance for one rejected generated response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preview: str = Field(max_length=REJECTED_OUTPUT_PREVIEW_MAX_CHARS)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    char_count: int = Field(ge=0)
    truncated: bool


def build_request_bound_autograder_schema(
    asset_id: str,
    scenario_id: str,
    rubric_version: str,
) -> dict[str, Any]:
    """Bind identity fields in the response schema to one exact development request."""

    schema = copy.deepcopy(AutograderResponse.model_json_schema())
    properties = schema.get("properties")

    if not isinstance(properties, dict):
        raise ValueError("Autograder response schema is missing object properties.")

    properties["asset_id"] = {
        "title": "Asset Id",
        "type": "string",
        "enum": [asset_id],
    }
    properties["scenario_id"] = {
        "title": "Scenario Id",
        "type": "string",
        "enum": [scenario_id],
    }
    properties["rubric_version"] = {
        "title": "Rubric Version",
        "type": "string",
        "enum": [rubric_version],
    }
    properties["result_source"] = {
        "title": "Autograder Response Source",
        "type": "string",
        "enum": [AutograderResponseSource.VLM_OUTPUT.value],
    }

    return schema


def capture_rejected_output(raw_output: str) -> RejectedOutputProvenance:
    """Retain a bounded preview plus full-output digest for failed validation."""

    preview = raw_output[:REJECTED_OUTPUT_PREVIEW_MAX_CHARS]

    return RejectedOutputProvenance(
        preview=preview,
        sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        char_count=len(raw_output),
        truncated=len(raw_output) > REJECTED_OUTPUT_PREVIEW_MAX_CHARS,
    )
