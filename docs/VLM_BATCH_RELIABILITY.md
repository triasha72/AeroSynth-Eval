# VLM batch reliability

## Milestone

AeroSynth-Eval v0.11 adds the first reliability layer around the development
batch runner introduced in v0.10.

The goal is to make repeated local development evaluation observable and
recoverable enough to support later evaluator validation.

## Shared MLX-VLM session

The real batch initializes MLX-VLM once, retaining the loaded model, processor,
model configuration, generation function, and chat-template adapter across all
12 development cases.

The structured JSON decoder is intentionally rebuilt per case. This preserves
case independence while avoiding repeated model loading.

## Typed failure taxonomy

The batch persists a failure category rather than only a free-form error string.

| Failure kind | Meaning | Retried? |
| --- | --- | --- |
| `model_load_failure` | local model/session could not initialize | No |
| `inference_failure` | generation raised a runtime error | Yes, bounded |
| `empty_response` | generation returned no usable text | No |
| `json_parse_failure` | output was not one exact JSON object | No |
| `schema_validation_failure` | JSON violated the frozen response schema | No |
| `request_binding_failure` | response disagreed with the exact request | No |
| `preparation_failure` | asset/request/provenance preparation failed | No |

## Retry policy

Retries apply only to transient inference failures.

The default is:

```text
max_retries = 1
```

The accepted range is 0 through 3.

This avoids converting deterministic evaluator or data-contract failures into
repeated model calls.

## Reliability provenance

Each case records:

- attempt count;
- elapsed seconds;
- final success/failure status;
- typed failure kind when unsuccessful;
- free-form diagnostic text when unsuccessful; and
- the complete validated single-case run record when successful.

The batch records:

- whether inference occurred;
- whether one shared local session was reused;
- session initialization time;
- total elapsed time;
- the retry policy; and
- all per-case records.

## Model-load failures

Model initialization is a batch-level dependency.

If it fails, AeroSynth-Eval records all scheduled development cases as
`model_load_failure` with `attempt_count = 0`. This distinguishes a batch that
could not begin inference from a batch where individual cases failed later.

## Scope

These additions measure execution reliability.

They do not establish:

- evaluator accuracy;
- human agreement;
- calibration;
- model superiority;
- inspection validity; or
- protected-test performance.

The next research milestones should use successful development batch outputs
together with independent human annotations before model-quality claims are
introduced.
