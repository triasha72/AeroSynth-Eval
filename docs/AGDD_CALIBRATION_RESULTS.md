# AGDD sequence-confidence calibration

Date: 2026-08-25

Confidence is the geometric mean probability assigned to the greedily generated output tokens.
Correctness is exact multilabel set match. Both models were evaluated on the same 22 AGDD held-out
validation examples.

| Model | Exact match | Mean confidence | ECE (5 bins) | Brier | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-VL-3B | 4.55% | 0.706 | 0.661 | 0.486 | 1.223 |
| Completion-only QLoRA | 22.73% | 0.664 | 0.437 | 0.378 | 0.974 |

Completion-only tuning improves every reported calibration metric, but the tuned model remains
substantially overconfident. These numbers are validation diagnostics, not protected-test estimates,
because the same 22-example official validation split was used for checkpoint selection. A frozen
selection/test subdivision is required for the final claim.
