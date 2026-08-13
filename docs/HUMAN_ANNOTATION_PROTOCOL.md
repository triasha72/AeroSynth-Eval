# Human Annotation Protocol v0.1

## Purpose

This protocol defines a small, development-only calibration exercise for the
procedural corpus. It establishes a reproducible human-rating interface before
any VLM evaluator is tuned or compared with people.

The queue contains 12 generated assets. It is balanced across the three aircraft
regions, four requested surface conditions, and four capture profiles. It does
not contain any protected test scenarios.

## Scope and limits

- This is an independent research prototype using synthetic imagery only.
- Raters assess visible evidence against the requested scenario; they do not
  diagnose defects, estimate severity, recommend maintenance, or determine
  aircraft airworthiness.
- The bundled queue is an annotation design artifact, not a source of labels,
  agreement results, calibration results, or VLM performance claims.
- Protected test scenarios must remain unseen by prompt-design and development
  workflows until a preregistered final evaluation is run.

## Materials

- Queue: `data/annotations/v0_1_development_annotation_queue.csv`
- Image assets: paths in the queue, relative to `data/`
- Rubric: `docs/ANNOTATION_GUIDE.md`
- Empty response schema: `data/annotations/v0_1_rater_response_template.csv`

Validate the queue before distributing it:

```bash
aerosynth-eval validate-annotation-queue \
  data/annotations/v0_1_development_annotation_queue.csv
```

## Rater procedure

1. Give each rater a separate copy of the response template and the same queue.
2. Use a pseudonymous ID such as `rater_a`; do not put names or contact details
   in the response file.
3. For every queued scenario, inspect the image and score each fixed dimension
   from 0 to 4 using the rubric anchors.
4. Record an `accept`, `reject`, or `uncertain` decision and concise, visible
   evidence for the decision and every score.
5. Work independently: raters must not see another rater's scores, rationales,
   VLM output, or proposed consensus while making their first-pass judgments.
6. Validate each complete submission before any later aggregation or adjudication.

Store raw submissions under `data/annotations/private/`; that directory is
ignored by Git. Do not commit raw rater identifiers or private responses to the
public repository.

```bash
mkdir -p data/annotations/private
cp data/annotations/v0_1_rater_response_template.csv \
  data/annotations/private/rater_a_v0_1.csv

# Fill all 12 rows in the copied CSV, then validate it.
aerosynth-eval validate-rater-annotations \
  data/annotations/private/rater_a_v0_1.csv
```

## Response schema

Each row must include a pseudonymous `rater_id`, one queue `scenario_id`, the
fixed rubric version, four integer scores in the inclusive range 0–4, and a
rationale for every score. The validator rejects missing scenarios, duplicate
rater/scenario pairs, incompatible queue or rubric versions, out-of-range
scores, and test-split scenarios.

The project makes no inter-rater agreement claim until at least two independent,
complete submissions have been collected and analyzed with a documented method.

## Future use

This development-only batch supports calibration of the rubric and rater
instructions. It must not be used to tune against protected test labels. A later
milestone can add anonymized consensus labels, agreement analysis, evaluator
comparisons, confidence calibration, and error slices under a fixed hold-out
protocol.
