# Development VLM batch evaluation

## Purpose

AeroSynth-Eval v0.10 adds a development-only VLM batch runner built on top of
the validated single-image MLX-VLM execution path.

The batch runner exists to establish reproducible multi-case execution,
provenance, failure isolation, and operational reliability before model-quality
evaluation begins.

It does not establish evaluator accuracy, human agreement, calibration, model
superiority, or protected-test performance.

## Development queue

The initial batch uses the existing frozen human-annotation queue:

`data/annotations/v0_1_development_annotation_queue.csv`

The queue contains exactly 12 development assets and is validated against the
frozen scenario matrix and asset registry before any inference begins.

Protected test assets are rejected before batch inference begins.

## Failure isolation

One failed case does not terminate the remaining batch.

Each case is stored as either:

- `success`, with its complete validated single-case provenance record; or
- `failed`, with an error and no accepted run record.

The v0.10 milestone deliberately uses only coarse `success` and `failed`
outcomes. Detailed failure taxonomy is deferred to a later reliability milestone.

## Dry run

```bash
aerosynth-eval run-vlm-batch --dry-run
```

A valid plan must report `inference_performed: false`,
`performance_claim_supported: false`, `split: development`, and 12 unique
development asset IDs.

## Metrics in this milestone

The batch summary reports operational quantities only:

- cases attempted;
- cases succeeded;
- cases failed; and
- execution success rate.

These quantities measure execution reliability, not evaluator quality.

## Current implementation limitation

The v0.10 batch runner intentionally prioritizes correctness and reuse of the
existing validated single-case execution path. Persistent shared model sessions,
parallel workers, retries, caching, resume support, detailed failure categories,
and distributed execution are deferred to later milestones.

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
synthetic, or self-generated data. It is not an operational inspection,
maintenance, airworthiness, defect-diagnosis, or certification system.
