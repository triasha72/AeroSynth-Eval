# PR #27 — Preference-supervised VLM LoRA/QLoRA adaptation

## Purpose

Prepare leakage-controlled pairwise preference training data and run VLM LoRA/QLoRA adaptation with MLX-VLM on Apple Silicon.

## Methodology boundary

- Do not claim results that have not actually been run.
- Keep GenAI-Bench/RichHF/public data out of AeroSynth's internal protected-test split.
- Do not fabricate human annotations or adjudications.
- Record model, dataset, prompt and code revisions for every reported result.

## Gate

COMPUTE/DATA GATE: prepare only TRAIN partition for fitting. Use VALIDATION for model/prompt selection and HELDOUT only for final comparison. Do not train on heldout labels.
