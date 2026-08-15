# PR #20 — Resumable cached sharded evaluation

## Purpose

Add resumable, cacheable, deterministic-shard infrastructure and reproducible experiment manifests for larger evaluation runs.

## Methodology boundary

- Do not claim results that have not actually been run.
- Keep GenAI-Bench/RichHF/public data out of AeroSynth's internal protected-test split.
- Do not fabricate human annotations or adjudications.
- Record model, dataset, prompt and code revisions for every reported result.

## Gate

No special external gate beyond passing unit/static checks and the smoke test described in README_APPLY.
