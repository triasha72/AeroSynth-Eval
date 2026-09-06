# AeroSynth-Eval

[Portfolio case study](https://triasha72.github.io/Portfolio/case-aerosynth.html)

[![CI](https://github.com/triasha72/AeroSynth-Eval/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/AeroSynth-Eval/actions/workflows/ci.yml)

AeroSynth-Eval looks at a difficult computer-vision problem: aircraft inspection
images are scarce, but a synthetic image is useful only if it helps on real
inspection photographs. This repository tests that idea instead of assuming it.

## Research question

Can a model recognize the requested aircraft region and visible condition, and
can it spot cases that should be rejected or reviewed by a person?

The real-image benchmark uses the public AGDD aircraft glass-canopy dataset.
Generated images appear only as clearly marked controls or training
augmentation. In the latest ten-seed experiment, mixed training improved average
macro F1 but reduced crack recall. That is an interesting trade-off, not a safety
win, and the roadmap reflects it.

## Project story

**Situation.** Real aircraft inspection images are difficult to collect, which
makes synthetic augmentation attractive. The risk is that generated images make
an average metric look better while weakening detection of an important defect.

**Task.** I set up a controlled test of whether synthetic data actually helps on
untouched real aircraft images.

**Action.** I audited the public AGDD canopy dataset, kept its 22-pair validation
set protected, and compared real-only, synthetic-only, and mixed training under
the same sample budget across ten seeds. I reported macro F1 and crack recall
separately and kept the VLM and human-rating work behind their own evidence
boundaries.

**Result.** Mixed training improved mean macro F1 from `0.3881` to `0.4419`, but
mean crack recall fell from `0.4000` to `0.3167`. I did not call that a win. The
experiment narrowed the next question to augmentation that helps overall
classification without sacrificing the protected defect metric.

## Scope

The v0.1 rubric evaluates:

- aircraft-region and inspection-context fidelity;
- requested surface-condition fidelity;
- visual quality, including focus, lighting, framing, glare, and occlusion;
- suitability for a non-operational inspection-data workflow.

All examples in this project must use public, generated, or self-created data.

## Safety and limitations

This is an independent research prototype. It is not an airworthiness,
maintenance-release, defect-diagnosis, or certification system. Its outputs support
research dataset quality review and must not replace qualified human inspection.

## What was built and why

The project builds an evaluation path from frozen synthetic scenarios to
auditable model and human comparisons. It combines typed evaluator contracts,
local and batched VLM execution, human-annotation ingestion, agreement and
adjudication workflows, public preference benchmarks, pairwise and multi-judge
evaluation, confidence calibration, human-alignment analysis,
preference-supervised adaptation contracts, experiment selection, and
deterministic distributed execution.

These layers were separated so software fixtures cannot be mistaken for human
or model evidence. The repository contains no genuine completed human-rating
set, adjudicated gold reference, aggregate protected-test VLM result, or
demonstrated fine-tuning improvement. The bundled images are simple synthetic
test assets—not operational inspection imagery or physical simulations.

### Capability map

| Layer | Implemented boundary |
|---|---|
| Benchmark integrity | 48 frozen scenarios, 48 checksummed assets, and a 36/12 development/protected-test split |
| Model execution | Shared-session MLX-VLM and free Kaggle/P100 Transformers development batches |
| Human workflow | Rater templates, ingestion, agreement analysis, manual adjudication, and human-alignment contracts |
| Judge evaluation | GenAI-Bench/RichHF adapters, pairwise and multi-judge comparison, preference metrics, and calibration |
| Adaptation | Leakage-controlled preference-data preparation and MLX-VLM LoRA/QLoRA training interfaces |
| Experiment systems | Registry-based comparison, explicit selection policy, deterministic sharding, caching, and reduction |

The [project notes](docs/) state the data, compute, and human-review gates for
each layer. A merged implementation is not treated as a measured evaluator win.

### Public human-preference evidence

The pinned GenAI-Bench image-generation `test_v1` parquet has now been
materialized through the real adapter. It contains 1,735 human pairwise votes
and 3,470 paired images. Prompt-group hashing produces 1,246 training, 254
validation, and 235 held-out examples without placing related pairs in
different partitions. The text-free summary and source checksum are in
`artifacts/genai_bench_human_votes_v1.json`; prompts and images remain outside
Git under the source's CC-BY-4.0 terms.

This closes the public human-vote ingestion gap for general generated-image
judging. It does not establish aircraft-inspection correctness or replace two
independent raters on the blinded aerospace development set.

The transfer release gate now protects crack recall explicitly. Although mixed
training improved mean macro F1 by `0.0537`, it reduced crack recall by `0.0833`,
outside the allowed `0.05` non-inferiority margin. The 44-image real evaluation
set also misses the 200-image minimum, and independent aerospace review is not
complete. The checked-in decision therefore remains `blocked` in
`reports/agdd_transfer_release_v1.json`.

## Benchmark foundation

The bundled manifest at `data/examples/v0_1_manifest.jsonl` contains six synthetic
scenario records: four development examples and two held-out test examples. It is
intentionally metadata-only, separate from the procedural PNG corpus and future
human-rating data.

## Real aircraft-domain transfer benchmark

A separate AGDD track now audits and evaluates 219 real paired-illumination
aircraft glass-canopy samples under the dataset's CC BY-NC-SA 4.0 license. A
transparent multilabel baseline trained on the official 197-pair training split
achieved macro F1 `0.6108`, micro F1 `0.5926`, and exact-match accuracy `0.1818`
on all 22 held-out validation pairs. This result is intentionally separate from
the synthetic protected test and is not an airworthiness or maintenance-release
claim. See `docs/AGDD_REAL_IMAGE_BASELINE.md` and the checksummed reports in
`reports/`.

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

The [volunteer study guide](docs/UNPAID_REVIEW_STUDY.md) and pilot builder create
two separate blank 12-case development files. This allows an unpaid rubric
pilot while protected-test claims remain blocked until genuine independent
reviews exist.

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

## VLM-ready autograder contract

The v0.8 contract creates deterministic, development-only prompt previews from
the frozen scenario matrix and asset registry. It also validates the shape and
metadata binding of a future evaluator response before any analysis is allowed.
The command never loads a model or performs inference, and it rejects every
protected test asset. The bundled response is a clearly marked synthetic
contract fixture, not a VLM output or evaluation result. Read
[the VLM autograder-contract note](docs/VLM_AUTOGRADER_CONTRACT.md) before
connecting any model runner.


## Local MLX-VLM development runner

The v0.9 runner is an optional Apple Silicon local-inference integration. It
accepts one generated development asset, rejects every protected test asset,
captures model and image provenance, and saves a local run record outside Git.
Its dry-run mode performs no model import, download, or inference; an actual
single response remains an individual research artifact, not an evaluator-quality
result. Read [the local-runner note](docs/MLX_VLM_DEVELOPMENT_RUNNER.md) before
installing MLX-VLM or running the command.

## Free Kaggle VLM batch

The Linux/CUDA backend loads Qwen2-VL once in 4-bit mode and reuses it across
the same request-bound 12-case development queue. The checked-in
[Kaggle notebook](notebooks/kaggle_vlm_batch.ipynb) records model, prompt,
image, response, failure, and timing provenance, then packages the ignored run
artifacts with a SHA-256 digest. See the [Kaggle runbook](docs/KAGGLE_VLM_BATCH.md).
The first measured P100 run attempted all 12 cases: eight responses passed the
strict schema and four JSON failures were retained in the frozen report. This
is execution-reliability evidence only; it is not human-alignment,
protected-test, or evaluator-performance evidence.

## Evaluation and scaling layers

The later evaluation layers are organized around explicit boundaries:

- public preference datasets stay separate from AeroSynth's internal protected
  test split;
- real human-agreement reports require two genuine independent submissions;
- adjudication is manual and never auto-filled from software fixtures;
- human-alignment analysis requires a completed adjudicated reference and real
  evaluator outputs;
- preference adaptation fits only on the training partition, uses validation
  for selection, and reserves held-out data for final comparison; and
- distributed runs preserve model, dataset, prompt, code, shard, and cache
  provenance before deterministic reduction.

See the focused notes for
[preference benchmarking](docs/PR17_PREFERENCE_BENCHMARK.md),
[human agreement](docs/PR24_HUMAN_AGREEMENT.md),
[human alignment](docs/PR26_HUMAN_ALIGNMENT.md),
[evaluator adaptation](docs/PR27_EVALUATOR_ADAPTATION.md),
[experiment selection](docs/PR28_EXPERIMENT_HILLCLIMBING.md), and
[distributed execution](docs/PR29_DISTRIBUTED_EXECUTION.md).

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

aerosynth-eval preview-autograder-prompt \
  asset-fuselage-corrosion-close-diffuse
aerosynth-eval validate-autograder-response \
  tests/fixtures/autograder/synthetic_valid_response.json

aerosynth-eval run-mlx-vlm-smoke \
  asset-fuselage-corrosion-close-diffuse \
  --dry-run

aerosynth-eval run-vlm-batch --backend transformers --dry-run
```

GitHub Actions runs formatting, linting, strict type checking, tests, and the
repository's synthetic smoke validations for pushes and pull requests targeting
`main`. CI validates software behavior and frozen inputs; it does not convert
synthetic fixtures into human, model-quality, or operational evidence.
