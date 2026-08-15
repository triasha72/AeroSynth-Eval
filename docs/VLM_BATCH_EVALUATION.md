# Development VLM batch evaluation

## Purpose

AeroSynth-Eval v0.11 provides a development-only VLM batch runner built on top
of the validated single-image MLX-VLM execution path.

The batch runner establishes reproducible multi-case execution, provenance,
failure isolation, typed reliability diagnostics, and bounded retry behavior
before model-quality evaluation begins.

It does not establish evaluator accuracy, human agreement, calibration, model
superiority, or protected-test performance.

## Development queue

The batch uses the existing frozen development annotation queue:

`data/annotations/v0_1_development_annotation_queue.csv`

The queue contains exactly 12 development assets and is validated against the
frozen scenario matrix and asset registry before inference begins.

Protected test assets are rejected before batch inference begins.

## Shared local session

For a real local batch, the MLX-VLM model and processor are initialized once and
reused across the 12 development cases.

A fresh JSON-schema logits processor is still constructed for every case so
structured-decoding parser state is never reused across independent responses.

The validated single-case path remains authoritative for:

- development-only request construction;
- image and provenance checks;
- structured output generation;
- strict JSON parsing;
- Pydantic response validation; and
- asset/scenario/rubric request binding.

## Failure isolation and taxonomy

One case failure does not terminate later cases.

Failed cases are classified as:

- `model_load_failure`
- `inference_failure`
- `empty_response`
- `json_parse_failure`
- `schema_validation_failure`
- `request_binding_failure`
- `preparation_failure`

A global model/session initialization failure records all 12 scheduled cases as
`model_load_failure` with zero fake inference attempts.

## Retry policy

Retries are deliberately narrow.

Only `inference_failure` is retryable. JSON failures, schema failures,
request-binding failures, preparation failures, and empty responses are not
blindly retried.

`--max-retries` is bounded from 0 through 3 and defaults to 1.

## Timing provenance

The batch record includes:

- shared-session setup time;
- per-case elapsed time;
- per-case attempt count; and
- total batch elapsed time.

These are operational measurements only and are not model-quality metrics.

## Dry run

```bash
aerosynth-eval run-vlm-batch --dry-run
```

A valid plan reports:

- `inference_performed: false`
- `performance_claim_supported: false`
- `shared_session_enabled: true`
- `split: development`
- the bounded retry policy
- 12 unique development assets

## Real development batch

After v0.11 is merged and local validation passes:

```bash
aerosynth-eval run-vlm-batch \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --max-tokens 800 \
  --temperature 0.0 \
  --max-retries 1
```

Batch records are written under:

`outputs/vlm_batches/`

The `outputs/` tree remains ignored by Git.

## Operational summary

The batch summary reports:

- cases scheduled;
- cases attempted;
- cases succeeded;
- cases failed;
- total inference attempts;
- retry attempts;
- execution success rate;
- typed failure counts;
- session reuse;
- session setup time; and
- total batch elapsed time.

These quantities measure execution reliability only.

## Deferred capabilities

The v0.11 runner remains sequential. The following are intentionally deferred:

- parallel workers;
- response caching;
- resume/checkpoint support;
- distributed execution; and
- evaluator-quality comparison against human reference labels.

## Validation

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/aerosynth_eval
python -m pytest -q
aerosynth-eval run-vlm-batch --dry-run
git diff --check
git status --short
```

## Safety boundary

AeroSynth-Eval remains an independent research prototype using public,
synthetic, or self-generated data.

It is not an operational inspection, maintenance, airworthiness,
defect-diagnosis, or certification system.
