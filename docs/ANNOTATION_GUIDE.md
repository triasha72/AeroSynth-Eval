# AeroSynth-Eval Annotation Guide v0.1

## Goal

Rate whether a synthetic image satisfies its requested inspection scenario using
the fixed 0–4 rubric. Annotators judge visible evidence, not whether an aircraft is
safe to operate.

## Required output

For every example, record:

- one score for `context_fidelity`;
- one score for `condition_fidelity`;
- one score for `image_quality`;
- one score for `inspection_utility`;
- an `accept`, `reject`, or `uncertain` decision; and
- a brief evidence-based rationale for each score.

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

1. Read the scenario specification before inspecting the image.
2. Score each dimension independently from visible evidence.
3. Write concise rationales that name the evidence or limitation.
4. Do not change a held-out test label after seeing model output.
5. For the human-alignment study, two raters independently label an overlapping
   subset before disagreements are adjudicated.

## Exclusions

Do not diagnose defects, estimate severity, infer maintenance actions, or make any
airworthiness or certification judgment.
