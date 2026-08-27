# AeroSynth-Eval evidence roadmap

[Project overview and measured results](README.md)

## Implemented engineering foundation

- [x] frozen 48-scenario procedural benchmark and protected split
- [x] request-bound evaluator contracts and rejected-output provenance
- [x] shared-session MLX-VLM development batch
- [x] shared-session 4-bit Transformers batch for a free Kaggle P100
- [x] first measured Kaggle/P100 development batch frozen with bounded failure evidence
- [x] bounded retries and typed execution failures
- [x] deterministic sharding, cache keys, checkpoints, and strict reduction primitives
- [x] human annotation, agreement, adjudication, alignment, and calibration contracts
- [x] leakage-controlled preference-adaptation and experiment-selection contracts

## Evidence still requiring real execution or people

- [ ] collect two genuine independent human-rating files
- [ ] adjudicate genuine disagreement cases
- [ ] evaluate VLM outputs against the adjudicated development reference
- [ ] run calibration and human-alignment reports
- [ ] execute preference adaptation and compare it on untouched held-out data
- [ ] publish limitations, seeds, revisions, hashes, variance, and regressions

The first measured P100 batch attempted all 12 development cases in one model
session. Eight responses passed the strict schema and four JSON failures were
retained in `reports/kaggle_vlm_batch_v0_1.json`. This closes the execution
gap, but it does not establish evaluator accuracy or human alignment.

Software fixtures and CI results cannot satisfy the remaining evidence gates.
