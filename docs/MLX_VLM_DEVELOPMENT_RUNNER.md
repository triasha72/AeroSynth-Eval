# Local MLX-VLM development runner

## Purpose

The v0.9 runner connects one local MLX-VLM inference to the existing v0.8
autograder contract. It accepts exactly one generated **development** asset,
checks the complete corpus and image digest first, attaches the image to a
runtime-specific prompt, and then requires one exact structured response.

The v0.9.2 runtime uses JSON-schema constrained decoding based on the frozen
`AutograderResponse` contract. The generated response is still passed through
the existing strict JSON parser, Pydantic schema validation, and request-binding
checks before a run record can be accepted.

It is a local smoke-run and provenance feature—not a benchmark, model-selection
workflow, or evaluator-performance result.

## Boundaries enforced by code

- Every protected test asset is rejected before a model can be loaded.
- The image must be part of the frozen registry and match its recorded SHA-256
  digest.
- The prompt preserves the asset/scenario IDs, rubric version, four fixed
  dimensions, and research-only safety boundary from v0.8.
- Actual local inference uses the frozen `AutograderResponse` JSON schema to
  constrain generation toward one schema-compatible JSON object.
- The returned text must still parse as one bare JSON object with
  `result_source: "vlm_output"`.
- Markdown, prose, missing fields, extra fields, invalid scores, and metadata
  mismatches are rejected rather than repaired after generation.
- The validated response must match the exact asset ID, scenario ID, and rubric
  version of the development request.
- A saved local record includes the model identifier, installed MLX-VLM version,
  runtime settings, prompt and image digests, raw output, and validated response.
- The runner reports no aggregate scores, agreement, calibration, accuracy, or
  model-comparison result.

Structured decoding improves response-format reliability only. It does not imply
that the evaluator is accurate, calibrated, human-aligned, or suitable for an
operational inspection workflow.

## Install the optional local dependency

MLX-VLM is optional because it is for local Apple Silicon execution and is not
installed in Linux GitHub Actions CI.

The v0.9.2 local runtime pins MLX-VLM to version `0.5.0` so the structured-output
integration is reproducible.

```bash
conda activate aerosynth-eval-py312
python -m pip install -e ".[dev,mlx]"