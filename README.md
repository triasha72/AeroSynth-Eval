# AeroSynth-Eval

[![CI](https://github.com/triasha72/AeroSynth-Eval/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/AeroSynth-Eval/actions/workflows/ci.yml)

AeroSynth-Eval is an independent research and engineering project exploring how
visual-language models can evaluate synthetic aerospace inspection imagery against
explicit, human-reviewable rubrics.

## Research question

Can an evaluator determine whether a generated aircraft-exterior inspection image
matches its requested context and condition, while identifying when the image should
be rejected or escalated for human review?

## Scope

The v0.1 rubric evaluates:

- aircraft-region and inspection-context fidelity;
- requested surface-condition fidelity;
- visual quality, including focus, lighting, framing, glare, and occlusion;
- suitability for a non-operational inspection-data workflow.

All examples in this project must use public, synthetic, or self-generated data only.

## Safety and limitations

This is an independent research prototype. It is not an airworthiness,
maintenance-release, defect-diagnosis, or certification system. Its outputs support
research dataset quality review and must not replace qualified human inspection.

## Current status

v0.3 freezes a balanced scenario design before image generation or model work.
It adds a protected 36/12 development/test split across aircraft regions, requested
surface conditions, lighting/viewpoint profiles, and controlled quality challenges.
It does not claim VLM evaluation accuracy, human agreement, calibration, or
fine-tuning results.

## Benchmark foundation

The bundled manifest at `data/examples/v0_1_manifest.jsonl` contains six synthetic
scenario records: four development examples and two held-out test examples. It is
intentionally metadata-only—no image assets or labels are distributed yet. This
separates data-contract validation from future image generation and human-rating
work.

Read [the dataset card](docs/DATASET_CARD.md) and
[the annotation guide](docs/ANNOTATION_GUIDE.md) before adding benchmark data.

## Scenario design

The frozen v0.1 matrix at `data/design/v0_1_scenario_matrix.csv` defines 48
metadata-only scenarios. Each aircraft-region/condition combination has four
capture profiles and exactly one protected test scenario. Read
[the scenario-design note](docs/SCENARIO_DESIGN.md) before generating any imagery.

## Development

```bash
conda activate aerosynth-eval-py312
python -m pip install -e ".[dev]"

python -m ruff format .
python -m ruff check .
python -m mypy src/aerosynth_eval
python -m pytest

aerosynth-eval info
aerosynth-eval rubric
aerosynth-eval validate-manifest data/examples/v0_1_manifest.jsonl
aerosynth-eval validate-scenario-matrix data/design/v0_1_scenario_matrix.csv
```

GitHub Actions runs formatting, linting, type checking, tests, and manifest
validation for pushes and pull requests targeting `main`.
