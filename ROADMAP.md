# AeroSynth-Eval evidence roadmap

## Implemented engineering foundation

- [x] frozen 48-scenario procedural benchmark and protected split
- [x] request-bound evaluator contracts and rejected-output provenance
- [x] shared-session MLX-VLM development batch
- [x] shared-session 4-bit Transformers batch for a free Kaggle P100
- [x] bounded retries and typed execution failures
- [x] deterministic sharding, cache keys, checkpoints, and strict reduction primitives
- [x] human annotation, agreement, adjudication, alignment, and calibration contracts
- [x] leakage-controlled preference-adaptation and experiment-selection contracts

## Evidence still requiring real execution or people

- [ ] run and freeze the Kaggle development batch record
- [ ] collect two genuine independent human-rating files
- [ ] adjudicate genuine disagreement cases
- [ ] evaluate VLM outputs against the adjudicated development reference
- [ ] run calibration and human-alignment reports
- [ ] execute preference adaptation and compare it on untouched held-out data
- [ ] publish limitations, seeds, revisions, hashes, variance, and regressions

Software fixtures and CI results cannot satisfy these evidence gates.
