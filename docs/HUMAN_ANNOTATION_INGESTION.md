# Human annotation ingestion

## Milestone

AeroSynth-Eval v0.13 introduces the development-only human annotation ingestion
workflow.

The previous evaluator-output milestone established a 12/12 structured-response
execution result on the fixed development queue. v0.13 therefore moves the
project from evaluator plumbing toward a real human-reference workflow.

This milestone does not compute human agreement. Agreement analysis is deferred
until a later PR after two real independent submissions exist.

## Why this layer exists

The repository already contains:

- a frozen 12-case development annotation queue;
- a fixed v0.1 rubric;
- strict CSV contracts for rater annotations; and
- validation that every rater must cover every queued scenario.

v0.13 adds the study-level workflow around those contracts.

## Private template generation

Generate one private CSV for each pseudonymous rater:

```bash
aerosynth-eval prepare-human-rater-template \
  rater_alpha \
  data/annotations/private/rater_alpha.csv

aerosynth-eval prepare-human-rater-template \
  rater_beta \
  data/annotations/private/rater_beta.csv
```

Each template:

- contains exactly 12 rows;
- is bound to the approved development queue;
- pre-populates annotation ID, rater ID, queue ID, scenario ID, and rubric
  version;
- leaves all ratings and rationales blank; and
- refuses to overwrite an existing file.

The `data/annotations/private/` directory is ignored by Git.

## Human collection protocol

Two actual raters should complete the templates independently.

Before both submissions are complete, raters should not see:

- VLM outputs;
- VLM scores or rationales;
- another rater's labels; or
- agreement statistics.

The software cannot prove that the study protocol was followed. It records this
limitation explicitly.

## Ingestion

After both private CSVs are complete:

```bash
aerosynth-eval ingest-human-annotations \
  data/annotations/private/rater_alpha.csv \
  data/annotations/private/rater_beta.csv
```

The command validates:

- exactly two source submissions;
- one pseudonymous rater per source;
- distinct rater IDs;
- non-synthetic-demo rater IDs;
- 12 records per rater;
- all 12 development scenarios per rater;
- queue identity;
- rubric identity;
- no duplicate scenario per rater;
- no duplicate annotation IDs across raters; and
- distinct source-file fingerprints.

## Ingestion manifest

Successful ingestion writes a manifest under:

```text
outputs/human_annotations/
```

The manifest includes only provenance/completeness metadata:

- manifest ID;
- study ID;
- timestamp;
- queue ID and queue SHA-256;
- development split;
- rubric version;
- two pseudonymous rater IDs;
- source basenames and SHA-256 digests;
- record counts; and
- explicit software-verification boundaries.

It does not copy:

- decisions;
- scores;
- rationales; or
- human agreement results.

## Explicit claim boundary

A successful v0.13 ingestion means:

```text
two structurally complete pseudonymous submissions
were validated against the frozen development queue
```

It does not prove:

- human authorship;
- independent rating behavior;
- evaluator accuracy;
- human agreement;
- calibration;
- model superiority; or
- operational inspection validity.

The manifest therefore records:

```text
human_authorship_verified_by_software = false
independence_verified_by_software = false
agreement_computed = false
human_agreement_claim_supported = false
```

## Validation

Before merging PR #16:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/aerosynth_eval
python -m pytest -q
aerosynth-eval info
aerosynth-eval prepare-human-rater-template --help
aerosynth-eval ingest-human-annotations --help
git diff --check
```

Do not fabricate completed rater submissions to make the CLI appear successful.

Use software fixtures only in tests. Real ingestion should happen only after
actual independent ratings are available.

## Next milestone

After two real submissions have been ingested, the next milestone should compute
human inter-rater agreement from those validated private inputs.

That agreement milestone should remain separate from v0.13 so ingestion,
agreement methodology, and eventual VLM-vs-human evaluation remain auditable.
