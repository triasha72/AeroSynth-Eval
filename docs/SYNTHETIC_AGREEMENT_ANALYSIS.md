# Synthetic Agreement-Analysis Fixture

## Purpose

This module is software-test infrastructure for the human-annotation pipeline.
It compares two deliberately simulated CSV fixtures in
`tests/fixtures/synthetic_demo/` and verifies that the project can compute
deterministic agreement summaries without exposing free-text rationales.

The fixture records are intentionally marked with both:

- `synthetic_demo_` rater IDs; and
- `Synthetic demo only:` at the beginning of every rationale.

The command rejects inputs that do not use both conventions.

## Run it

~~~bash
aerosynth-eval summarize-synthetic-agreement \
  tests/fixtures/synthetic_demo/synthetic_demo_rater_a.csv \
  tests/fixtures/synthetic_demo/synthetic_demo_rater_b.csv
~~~

Before summarizing, the command validates each file against the frozen
development-only annotation queue, scenario matrix, and asset registry.

## Output

The JSON summary contains:

- exact agreement for `accept`, `reject`, and `uncertain` decisions;
- exact and within-one agreement for each 0–4 rubric dimension;
- mean absolute score difference for each rubric dimension; and
- disagreement candidates, triggered by different decisions or a score
  difference of at least two points.

The command intentionally omits rationale text from its output.

## Boundaries

Every summary carries these fields:

~~~json
{
  "data_classification": "synthetic_demo_fixture",
  "claim_scope": "software_test_fixture_only",
  "human_agreement_claim_supported": false
}
~~~

The output is not evidence of:

- human-rater agreement, calibration, or consensus;
- VLM evaluation quality, alignment, or accuracy;
- a gold-label set or benchmark result; or
- aircraft condition, inspection suitability, maintenance, airworthiness, or
  certification.

Do not use the fixture CSVs or their numeric output in the README, CV,
benchmark reporting, model selection, or any claim about humans or models.

## Future human-study work

Actual independent human submissions remain private under
`data/annotations/private/` and must not be passed to this command. A future
human-study milestone should define its method before aggregation, retain the
development/test separation, and add a separate analysis path with appropriate
documentation and review.
