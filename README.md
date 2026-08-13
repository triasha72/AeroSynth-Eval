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

v0.6 adds a development-only human-annotation protocol to the reproducible
48-scenario procedural corpus. The bundled 12-image calibration queue has no
scores or consensus labels; it defines a controlled path for collecting
independent human ratings before any VLM evaluator is tuned or compared with
people. The assets remain deliberately simple synthetic test images—not
operational inspection imagery, physical simulations, or evidence of VLM
evaluation accuracy, human agreement, calibration, or fine-tuning.

## Benchmark foundation

The bundled manifest at `data/examples/v0_1_manifest.jsonl` contains six synthetic
scenario records: four development examples and two held-out test examples. It is
intentionally metadata-only, separate from the procedural PNG corpus and future
human-rating data.

Read [the dataset card](docs/DATASET_CARD.md) and
[the annotation guide](docs/ANNOTATION_GUIDE.md) before adding benchmark data.

## Scenario design

The frozen v0.1 matrix at `data/design/v0_1_scenario_matrix.csv` defines 48
metadata-only scenarios. Each aircraft-region/condition combination has four
capture profiles and exactly one protected test scenario. Read
[the scenario-design note](docs/SCENARIO_DESIGN.md) before generating any imagery.

## Procedural corpus baseline

The registry at `data/registry/v0_1_asset_registry.jsonl` records 48 generated
assets—one for every frozen scenario—along with renderer version, generation
specification, seed, UTC timestamp, and SHA-256 digest. Read
[the procedural-corpus note](docs/PROCEDURAL_CORPUS.md) and
[the image-generation protocol](docs/IMAGE_GENERATION_PROTOCOL.md) before producing
or reviewing imagery.

## Human annotation protocol

The development-only queue at
`data/annotations/v0_1_development_annotation_queue.csv` selects 12 balanced
generated assets for independent human rating. It intentionally excludes all
protected test assets and includes no labels or agreement results. Read
[the human-annotation protocol](docs/HUMAN_ANNOTATION_PROTOCOL.md) before
collecting or validating any response files.

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
aerosynth-eval validate-asset-registry data/registry/v0_1_asset_registry.jsonl
aerosynth-eval validate-materialized-corpus
aerosynth-eval validate-annotation-queue \
  data/annotations/v0_1_development_annotation_queue.csv
```

GitHub Actions runs formatting, linting, type checking, tests, and validation of
the bundled manifest, scenario design, provenance, materialized corpus, and
development annotation queue for pushes and pull requests targeting `main`.
