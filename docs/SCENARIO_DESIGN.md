# AeroSynth-Eval Scenario Design v0.1

## Purpose

This document freezes the metadata-only scenario design used before synthetic
image generation. It prevents prompt iteration or model selection from quietly
changing the held-out evaluation set.

## Factor design

The matrix contains 48 scenarios:

- 3 aircraft-exterior regions: fuselage panel, wing panel, and empennage panel;
- 4 requested conditions: no visible defect, corrosion, surface crack, and coating damage;
- 4 capture profiles per region/condition pair; and
- 36 development and 12 held-out test scenarios.

Each of the 12 region/condition pairs has exactly four scenarios and exactly one
test scenario. Each capture profile appears 12 times overall and three times in
the test split.

| Capture profile | Viewpoint | Lighting | Challenge |
| --- | --- | --- | --- |
| `close_diffuse` | Close inspection view | Diffuse daylight | None |
| `oblique_directional` | Oblique inspection view | Directional daylight | None |
| `close_glare` | Close inspection view | Directional daylight | Mild glare |
| `oblique_blur` | Oblique inspection view | Overcast daylight | Mild motion blur |

## Split policy

The test split is protected. Do not use test scenarios to choose image-generation
prompts, tune evaluator prompts, select a model, or set thresholds. Any future
v0.2 matrix must be created as a new versioned file rather than editing this one.

## What this matrix is—and is not

This is a design specification, not a labeled dataset and not proof that an image
generator can create realistic defects. It contains no image assets, human labels,
model outputs, or performance claims.

Future imagery must be synthetic, public, or self-generated and must retain
generator, model-version, prompt, seed, generation date, and provenance metadata.
No proprietary, customer, operational, or inspection-decision imagery belongs in
this project.

## Safety boundary

The scenarios support non-operational research on image and evaluator quality.
They do not diagnose defects, determine severity, recommend maintenance, or make
airworthiness, maintenance-release, or certification decisions.
