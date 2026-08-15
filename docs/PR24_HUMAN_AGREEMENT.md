# PR #24 — Human-human agreement analysis

## Purpose

Compute real human-human agreement only after two genuine complete rater submissions exist; unit tests use software fixtures only.

## Methodology boundary

- Do not claim results that have not actually been run.
- Keep GenAI-Bench/RichHF/public data out of AeroSynth's internal protected-test split.
- Do not fabricate human annotations or adjudications.
- Record model, dataset, prompt and code revisions for every reported result.

## Gate

HUMAN GATE: do not run the real report until rater_alpha.csv and rater_beta.csv contain genuine independent completed ratings. The code/tests may be merged before the real report, but make no human-agreement claim until the gate is satisfied.
