# Synthetic Demo Annotation Fixtures

These two CSVs contain intentionally simulated scores and rationales for exercising AeroSynth-Eval's CSV validation and future agreement-analysis code.

They are not human ratings, VLM outputs, gold labels, consensus labels, calibration evidence, or evaluation results. Do not use them in the README, CV, benchmark reporting, model selection, or any claim about human agreement or evaluator performance.

- synthetic_demo_rater_a.csv uses rater ID synthetic_demo_a.
- synthetic_demo_rater_b.csv uses rater ID synthetic_demo_b.
- Both cover the 12 development-only scenarios in v0_1_development_calibration.
- Each rationale begins with Synthetic demo only: to prevent accidental reuse as real evidence.

For repository tests, place these files under a clearly named path such as tests/fixtures/synthetic_demo/, not data/annotations/private/.
