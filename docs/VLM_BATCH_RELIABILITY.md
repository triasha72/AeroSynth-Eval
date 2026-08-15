# VLM batch reliability

## Milestone

AeroSynth-Eval v0.12 extends the development batch reliability layer with
request-bound structured generation and retained rejected-output provenance.

The batch runner remains focused on operational evaluation-pipeline reliability.
It does not compute evaluator-quality metrics.

## Shared MLX-VLM session

The real batch initializes MLX-VLM once, retaining the loaded model, processor,
model configuration, generation function, and chat-template adapter across all
12 development cases.

A fresh JSON-schema logits processor is constructed for every case.

In v0.12 that schema is also bound to the exact request identity before the
decoder is created.

## Request-bound identity fields

For every case, structured decoding pins:

- exact `asset_id`;
- exact `scenario_id`;
- exact `rubric_version`; and
- `result_source = vlm_output`.

This addresses identity drift observed during the first v0.11 real development
batch.

The strict response-binding validator remains active after generation.

## Typed failure taxonomy

The batch persists:

- `model_load_failure`
- `inference_failure`
- `empty_response`
- `json_parse_failure`
- `schema_validation_failure`
- `request_binding_failure`
- `preparation_failure`

Only `inference_failure` is retryable.

## Rejected-output provenance

For JSON, schema, and request-binding failures, the batch case can now retain a
bounded `rejected_output` object with:

- `preview`
- `sha256`
- `char_count`
- `truncated`

The batch summary additionally reports `rejected_outputs_retained`.

Rejected text remains diagnostic evidence and is never silently repaired.

## Retry policy

The default remains:

```text
max_retries = 1
```

The accepted range is 0 through 3.

Retries are deliberately limited to transient inference failures.

## Reliability provenance

Each case records:

- attempt count;
- elapsed seconds;
- final status;
- typed failure kind when unsuccessful;
- diagnostic error text;
- bounded rejected-output provenance when available; and
- complete validated single-case provenance when successful.

The batch records:

- whether inference occurred;
- whether the shared session was reused;
- whether request-bound generation was enabled;
- session initialization time;
- total elapsed time;
- retry policy; and
- all case records.

## Validation

Before merging v0.12:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/aerosynth_eval
python -m pytest -q
aerosynth-eval info
aerosynth-eval run-vlm-batch --max-retries 1 --dry-run
git diff --check
```

After merge, run one real development batch for regression comparison.

## Scope

AeroSynth-Eval remains an independent research prototype using public,
synthetic, or self-generated data.

It is not an operational inspection, maintenance, airworthiness,
defect-diagnosis, or certification system.

Request-bound generation and structured-response reliability do not establish
evaluator accuracy. Human-reference validation remains a later milestone.
