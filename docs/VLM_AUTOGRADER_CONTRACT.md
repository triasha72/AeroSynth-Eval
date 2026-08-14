# VLM-ready autograder contract

## Purpose

This v0.8 milestone defines the boundary between frozen AeroSynth-Eval metadata
and a future visual-language-model (VLM) runner. It creates a deterministic
prompt preview and validates one strict JSON response. It does not load a model,
send an image to a provider, or report evaluator performance.

## Development-only request grounding

`aerosynth-eval preview-autograder-prompt <asset_id>` resolves an asset through
the frozen registry and scenario matrix. The request includes the asset and
scenario IDs, image reference, component, requested condition, capture profile,
viewpoint, lighting, quality challenge, and fixed rubric version.

Only generated development assets are accepted. Every protected test asset is
rejected before a prompt can be previewed or a response can be validated.

## Prompt contents

The preview includes:

- a research-only scope boundary;
- the grounded development request;
- the fixed four-dimension rubric and 0–4 score anchors;
- a JSON schema for the required response; and
- an explicit notice that the command does not run model inference.

A future runner is responsible for attaching the referenced image and calling a
model. That runner must preserve this request metadata and return only the
structured response required by the schema.

## Response contract

Each response contains:

- `asset_id`, `scenario_id`, and `rubric_version`, which must exactly match the
  grounded request;
- `result_source`, either `synthetic_contract_fixture` or `vlm_output`;
- `decision`: `accept`, `reject`, or `uncertain`;
- `confidence` between 0 and 1;
- exactly one scored rationale for each of the four rubric dimensions; and
- a concise `summary`.

The response parser rejects extra fields, missing fields, invalid score ranges,
duplicate dimensions, omitted dimensions, request mismatches, and protected test
assets. CLI summaries intentionally omit rationale text and aggregate metrics.

## Synthetic test fixture

`tests/fixtures/autograder/synthetic_valid_response.json` is only a software-test
fixture. Its source and all rationales are explicitly marked. It is not a human
rating, VLM output, gold label, consensus label, calibration record, benchmark
result, or evidence of evaluator quality.

## Commands

```bash
aerosynth-eval preview-autograder-prompt \
  asset-fuselage-corrosion-close-diffuse

aerosynth-eval validate-autograder-response \
  tests/fixtures/autograder/synthetic_valid_response.json
```

Both commands are contract checks. Neither command performs model inference.

## Safety boundary

This is an independent research prototype using synthetic, public, or
self-generated data only. It is not an operational inspection, maintenance,
airworthiness, defect-diagnosis, or certification system. No VLM accuracy,
alignment, calibration, or human-agreement claim is supported by this milestone.
