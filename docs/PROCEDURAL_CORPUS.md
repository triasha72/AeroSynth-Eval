# Procedural Corpus Baseline

## Purpose

The v0.1 procedural corpus turns the frozen 48-scenario matrix into small,
deterministic PNG test assets. It exercises the end-to-end dataset, provenance,
checksum, and future autograder interfaces before an image-generation model is
introduced.

## What it is

- A programmatic renderer for generic aircraft-exterior panel illustrations.
- A controlled representation of the four requested surface conditions.
- A repeatable way to apply the four capture-profile challenges.
- A fully versioned asset registry with per-asset seed and SHA-256 evidence.

## What it is not

- It is not photorealistic aerospace inspection imagery.
- It is not an aircraft-maintenance, defect-detection, certification, or
  airworthiness system.
- It does not establish that a VLM performs well on real inspection images.

## Materialization

Run the following from the repository root after installing the project:

```bash
aerosynth-eval materialize-procedural-corpus
aerosynth-eval validate-materialized-corpus
```

The first command writes PNGs below `data/assets/v0_1/` and updates the registry
from `planned` to `generated`. The second command verifies that every generated
file is a PNG and matches its recorded SHA-256 digest.

## Reproducibility

Each asset seed is derived from its immutable `scenario_id`; rendering the same
scenario with the same renderer and Pillow versions yields the same image bytes.
Both versions are recorded in the asset registry. The UTC timestamp records when
the registry transitioned from a planned to a generated state. A changed renderer
version requires a new corpus version rather than silently rewriting an existing
benchmark.
