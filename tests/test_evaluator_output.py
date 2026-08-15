import hashlib

from aerosynth_eval.contracts import AutograderResponseSource
from aerosynth_eval.evaluator_output import (
    REJECTED_OUTPUT_PREVIEW_MAX_CHARS,
    build_request_bound_autograder_schema,
    capture_rejected_output,
)


def test_request_bound_schema_pins_identity_fields() -> None:
    schema = build_request_bound_autograder_schema(
        "asset-wing-crack-close-diffuse",
        "wing-crack-close-diffuse",
        "v0.1",
    )

    properties = schema["properties"]

    assert properties["asset_id"]["enum"] == ["asset-wing-crack-close-diffuse"]
    assert properties["scenario_id"]["enum"] == ["wing-crack-close-diffuse"]
    assert properties["rubric_version"]["enum"] == ["v0.1"]
    assert properties["result_source"]["enum"] == [AutograderResponseSource.VLM_OUTPUT.value]


def test_request_bound_schema_retains_evaluation_fields() -> None:
    schema = build_request_bound_autograder_schema(
        "asset-wing-crack-close-diffuse",
        "wing-crack-close-diffuse",
        "v0.1",
    )

    properties = schema["properties"]

    assert "decision" in properties
    assert "confidence" in properties
    assert "scores" in properties
    assert "summary" in properties


def test_capture_rejected_output_records_full_digest() -> None:
    raw = "```json\n{}\n```"

    provenance = capture_rejected_output(raw)

    assert provenance.preview == raw
    assert provenance.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert provenance.char_count == len(raw)
    assert provenance.truncated is False


def test_capture_rejected_output_caps_preview_without_changing_digest() -> None:
    raw = "x" * (REJECTED_OUTPUT_PREVIEW_MAX_CHARS + 37)

    provenance = capture_rejected_output(raw)

    assert len(provenance.preview) == REJECTED_OUTPUT_PREVIEW_MAX_CHARS
    assert provenance.preview == raw[:REJECTED_OUTPUT_PREVIEW_MAX_CHARS]
    assert provenance.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert provenance.char_count == len(raw)
    assert provenance.truncated is True
