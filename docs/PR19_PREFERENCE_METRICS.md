# PR #19 — Judge-quality metrics and position-bias controls

## Purpose

Measure judge-human agreement with class-balanced metrics, bootstrap uncertainty, tie analysis, and A/B order consistency.

## Methodology boundary

- Do not claim results that have not actually been run.
- Keep GenAI-Bench/RichHF/public data out of AeroSynth's internal protected-test split.
- Do not fabricate human annotations or adjudications.
- Record model, dataset, prompt and code revisions for every reported result.

## Gate

No special external gate beyond passing unit/static checks and the smoke test described in README_APPLY.
