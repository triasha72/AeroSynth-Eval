# Local MLX-VLM development runner

## Purpose

The v0.9 runner connects one local MLX-VLM inference to the existing v0.8
autograder contract. It accepts exactly one generated **development** asset,
checks the complete corpus and image digest first, attaches the image to a
runtime-specific prompt, and then requires one exact JSON response.

It is a local smoke-run and provenance feature—not a benchmark, model-selection
workflow, or evaluator-performance result.

## Boundaries enforced by code

- Every protected test asset is rejected before a model can be loaded.
- The image must be part of the frozen registry and match its recorded SHA-256
  digest.
- The prompt preserves the asset/scenario IDs, rubric version, four fixed
  dimensions, and research-only safety boundary from v0.8.
- The model must return one bare JSON object with `result_source: "vlm_output"`.
  Markdown, prose, missing fields, extra fields, invalid scores, and metadata
  mismatches are rejected.
- A saved local record includes the model identifier, installed MLX-VLM version,
  runtime settings, prompt and image digests, raw output, and validated response.
- The runner reports no aggregate scores, agreement, calibration, accuracy, or
  model-comparison result.

## Install the optional local dependency

MLX-VLM is optional because it is for local Apple Silicon execution and is not
installed in Linux GitHub Actions CI.

```bash
conda activate aerosynth-eval-py312
python -m pip install -e ".[dev,mlx]"
```

The first actual inference may download the selected model into the local model
cache. Start with the small 4-bit model configured by the command, and run only
one development asset in this milestone.

## Inspect the no-inference plan first

```bash
aerosynth-eval run-mlx-vlm-smoke \
  asset-fuselage-corrosion-close-diffuse \
  --dry-run
```

This validates the corpus, the selected development asset, image digest, prompt
binding, and runtime configuration. It does not import MLX-VLM, download a
model, create an output file, or run inference.

## Run one local smoke inference

```bash
aerosynth-eval run-mlx-vlm-smoke \
  asset-fuselage-corrosion-close-diffuse \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --max-tokens 800 \
  --temperature 0.0
```

The command writes one JSON provenance record under `outputs/mlx_vlm/`. That
directory is ignored by Git. Inspect the record locally; do not commit it or
describe a single response as an accuracy, alignment, calibration, or
human-agreement result.

## Validate the change

```bash
python -m ruff format .
python -m ruff check .
python -m mypy src/aerosynth_eval
python -m pytest -q

aerosynth-eval run-mlx-vlm-smoke \
  asset-fuselage-corrosion-close-diffuse \
  --dry-run
```

## Safety boundary

This is an independent research prototype using only public, synthetic, or
self-generated data. It is not an operational inspection, maintenance,
airworthiness, defect-diagnosis, or certification system. A successful schema
validation or a single local model response is not evidence of evaluator
quality.
