# PR #25 — Adjudicated development reference

## Purpose

Create a manual adjudication queue for human disagreements and validate the completed development reference without automatic label fabrication.

## Methodology boundary

- Do not claim results that have not actually been run.
- Keep GenAI-Bench/RichHF/public data out of AeroSynth's internal protected-test split.
- Do not fabricate human annotations or adjudications.
- Record model, dataset, prompt and code revisions for every reported result.

## Gate

HUMAN GATE: adjudication requires real disagreement cases. Never auto-fill adjudicated labels.
