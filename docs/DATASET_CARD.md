# AeroSynth-Eval v0.1 Manifest Card

## Purpose

This manifest defines the first versioned benchmark contract for AeroSynth-Eval.
It represents synthetic aircraft-exterior inspection-image scenarios and is used to
validate data structure before any model evaluation begins.

## Scope and safety

The benchmark supports research on synthetic image quality and rubric alignment.
It is not designed for airworthiness, defect diagnosis, maintenance release, or
certification. No project result may be used as an operational inspection decision.

## Contents

`data/examples/v0_1_manifest.jsonl` contains six metadata-only records:

- four development records;
- two held-out test records;
- synthetic provenance only;
- no distributed image assets; and
- no human labels yet.

Each record specifies the requested component, surface condition, viewpoint,
lighting, required visual attributes, split, provenance, and annotation status.

## Provenance policy

Future records must be public, synthetic, or self-generated. Every record requires
a provenance category and source note. Proprietary, controlled, customer, or
operational inspection imagery is out of scope.

## Intended annotation protocol

Human raters will assign one 0–4 score for each fixed rubric dimension and one
decision: `accept`, `reject`, or `uncertain`. The test split must remain untouched
during prompt iteration. Double-rating and adjudication will be introduced before
any claim about human agreement is made.

## Limitations

This fixture proves manifest validation only. It does not contain image files,
model outputs, annotator labels, agreement statistics, calibration results, or
fine-tuning data.
