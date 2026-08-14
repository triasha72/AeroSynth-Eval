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

v0.7 adds a deliberately simulated agreement-analysis fixture to the
development-only human-annotation protocol and reproducible 48-scenario
procedural corpus. It exercises validation, deterministic agreement summaries,
and disagreement reporting for clearly marked synthetic test records. It does
not produce human-rater agreement, calibration, consensus, VLM-evaluation, or
fine-tuning evidence. The assets remain deliberately simple synthetic test
images—not operational inspection imagery or physical simulations.

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

## Synthetic agreement-analysis fixture

The paired fixtures in `tests/fixtures/synthetic_demo/` are deliberately
simulated inputs for testing the agreement-analysis pipeline. They use
synthetic IDs and rationales, are checked in CI, and are never human ratings,
VLM outputs, gold labels, consensus labels, calibration evidence, or evaluation
results. Read [the synthetic agreement-analysis note](docs/SYNTHETIC_AGREEMENT_ANALYSIS.md)
before using the command. Do not use fixture output in the README, CV,
benchmark reporting, model selection, or any claim about human agreement or
evaluator performance.

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

aerosynth-eval validate-rater-annotations \
  tests/fixtures/synthetic_demo/synthetic_demo_rater_a.csv
aerosynth-eval validate-rater-annotations \
  tests/fixtures/synthetic_demo/synthetic_demo_rater_b.csv
aerosynth-eval summarize-synthetic-agreement \
  tests/fixtures/synthetic_demo/synthetic_demo_rater_a.csv \
  tests/fixtures/synthetic_demo/synthetic_demo_rater_b.csv
```

GitHub Actions runs formatting, linting, type checking, tests, and validation of
the bundled manifest, scenario design, provenance, materialized corpus, and
development annotation queue, as well as the synthetic demo fixtures, for pushes
and pull requests targeting `main`.
