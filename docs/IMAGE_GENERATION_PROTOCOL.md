# AeroSynth-Eval Image-Generation Protocol v0.1

## Purpose

This protocol governs production of the synthetic images reserved in
`data/registry/v0_1_asset_registry.jsonl`. Its goal is reproducible research
data, not photorealistic aircraft inspection, operational inspection, or defect
diagnosis.

## Before generation

1. Select one permitted source: synthetic, public, or self-generated imagery.
2. Record the generator name, model name, and model version.
3. Write the complete image-generation prompt from the frozen scenario record.
4. Select and record a non-negative integer seed.
5. Do not use the protected test split to choose prompts, models, image settings,
   rejection thresholds, or evaluation procedures.

## Required record after generation

Move an asset from `planned` to `generated` or `rejected` only after recording:

- the generator name, model, and version;
- the full generation prompt;
- the seed;
- an offset-aware generation timestamp;
- the SHA-256 checksum of the image file;
- the canonical path `assets/v0_1/<scenario_id>.png`; and
- a concise source note.

The asset-registry validator rejects generated or rejected records that omit any
of this evidence. It also rejects a registry that does not map one-to-one to the
frozen scenario matrix.

## Image review

Human reviewers should assess whether the image visibly matches the requested
scenario and whether it is suitable for non-operational research-data review.
They must not diagnose defects, estimate severity, prescribe maintenance, or make
airworthiness, maintenance-release, or certification judgments.

Do not tune an evaluator or change human labels after viewing held-out test model
outputs. Record rejected generations rather than silently replacing them.

## Current state

The v0.1 registry contains only `planned` records. No generator has been selected
and no image assets, human labels, or VLM results are distributed in this release.
