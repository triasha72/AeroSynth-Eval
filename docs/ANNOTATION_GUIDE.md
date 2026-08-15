# AeroSynth-Eval Annotation Guide v0.1

## Goal

Rate whether a synthetic image satisfies its requested inspection scenario using
the fixed 0–4 rubric.

Annotators judge visible evidence in a synthetic research image. They do not
judge whether an aircraft is safe to operate.

## Study boundary

The current human-alignment study uses only the frozen 12-case development
annotation queue.

Do not label protected-test assets during development methodology work.

Human raters should not be shown:

- VLM decisions;
- VLM scores;
- VLM rationales;
- another rater's labels; or
- agreement summaries

until both independent development submissions are complete.

## Pseudonymous rater IDs

Use a study pseudonym such as:

```text
rater_alpha
rater_beta
```

Do not put a person's name, email address, student ID, employee ID, or other
direct identifier in the annotation CSV.

The software validates the pseudonymous identifier format but cannot verify the
real-world identity or authorship of a rater.

## Required output

For every queued scenario, record:

- one `accept`, `reject`, or `uncertain` decision;
- one decision rationale;
- one score for `context_fidelity`;
- one rationale for `context_fidelity`;
- one score for `condition_fidelity`;
- one rationale for `condition_fidelity`;
- one score for `image_quality`;
- one rationale for `image_quality`;
- one score for `inspection_utility`; and
- one rationale for `inspection_utility`.

Every rater must complete all 12 development scenarios before the submission is
eligible for ingestion.

## Score scale

| Score | Meaning |
| --- | --- |
| 0 | Absent, unusable, or clearly contradictory |
| 1 | Weak evidence with major deficiencies |
| 2 | Partially satisfied with material limitations |
| 3 | Mostly satisfied with minor limitations |
| 4 | Fully satisfied and clearly usable |

## Decision rule

- Use `accept` only when the requested context and condition are visible and the
  image is sufficiently clear for research dataset review.
- Use `reject` when the image contradicts the scenario or is unusable because of
  severe blur, glare, framing, occlusion, or artifacting.
- Use `uncertain` when the evidence is ambiguous. Do not infer a condition that
  cannot be seen clearly.

## Annotation protocol

1. Use the approved development queue and the generated private template.
2. Read the scenario specification before inspecting the image.
3. Inspect only the corresponding synthetic image.
4. Score every dimension independently from visible evidence.
5. Write concise rationales that name visible evidence or a limitation.
6. Complete all 12 scenarios without viewing VLM outputs or another rater's
   submission.
7. Do not alter a completed submission after seeing VLM results or agreement
   statistics.
8. Submit the completed private CSV for ingestion validation.
9. Do not perform disagreement adjudication until the independent-rating phase
   is closed.

## Private-data handling

Recommended local paths are:

```text
data/annotations/private/rater_alpha.csv
data/annotations/private/rater_beta.csv
```

The repository `.gitignore` excludes `data/annotations/private/`.

Completed rating CSVs should therefore remain local/private and should not be
committed to Git.

PRs may contain:

- annotation software;
- blank protocols/templates;
- validation tests; and
- aggregate, de-identified results when scientifically appropriate.

PRs should not contain completed private rater submissions unless an explicit
data-governance decision has been made outside this tooling workflow.

## Independence boundary

AeroSynth-Eval can check:

- both submissions use distinct pseudonymous rater IDs;
- each rater covers all 12 scenarios;
- each scenario appears once per rater;
- queue and rubric identities are correct; and
- source-file fingerprints are distinct.

The software cannot verify that:

- a human actually authored the file;
- the rater worked independently;
- the rater did not see model output; or
- the rater followed institutional study requirements.

Those are study-protocol responsibilities rather than software guarantees.

## Exclusions

Do not diagnose defects, estimate severity, infer maintenance actions, or make
any airworthiness or certification judgment.

If the annotation work is conducted as part of a formal institutional study,
follow the applicable research-ethics and data-governance requirements for that
study.
