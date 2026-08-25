# AGDD Qwen2.5-VL-3B QLoRA result

Date: 2026-08-25

A real rank-8 QLoRA adapter was trained on AGDD's native aircraft-canopy defect task using a free
Colab Tesla T4. The official split was preserved: 197 training examples and 22 untouched validation
examples. This experiment does not remap AGDD labels into AeroSynth labels and does not touch the
AeroSynth protected test set.

## Training

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Quantization: 4-bit NF4 with double quantization
- Adapter targets: `q_proj`, `v_proj`
- Rank/alpha: 8/16
- Steps: 49, approximately one pass with batch 1 and four-step accumulation
- Runtime: 750 seconds
- Mean training loss: 11.835
- Saved adapter archive: 6.6 MiB

## Held-out comparison

| Model | Parse rate | Exact set match | Mean latency |
| --- | ---: | ---: | ---: |
| Base | 100% | 4.55% (1/22) | 563 ms |
| QLoRA adapter | 100% | 0% (0/22) | 1,309 ms |

The adapter is rejected. It overpredicted almost all labels: spot on 22/22 examples, contusion on
21/22, and both scratches and crack on 20/22. This is a genuine post-training regression, likely
caused by the tiny, multilabel, class-imbalanced training set plus an overly short single epoch.
The result proves that the VLM QLoRA pipeline executes end to end; it does not prove successful
domain adaptation.

## Balanced, checkpoint-selected iteration

A second 49-step run oversampled every training example containing the rare `crack` class by 7x and
evaluated all 22 held-out examples every 10 steps. TRL 0.28 rejects `assistant_only_loss=True` for
vision-language models, so this run retained full-sequence loss and records that limitation instead
of silently claiming assistant-only masking. Checkpoint 40 was selected by validation loss.

| Iteration | Best validation loss | Exact set match | Mean latency | Decision |
| --- | ---: | ---: | ---: | --- |
| Original QLoRA | not measured during training | 0% (0/22) | 1,309 ms | Reject |
| Balanced + checkpoint selection | 9.4666 at step 40 | 0% (0/22) | 1,377 ms | Reject |

The balanced run still collapsed toward predicting most labels: spot on 22/22 examples, contusion
on 21/22, crack on 20/22, and scratches on 16/22. It improved neither exact match nor latency and is
therefore also rejected. The next iteration should change the objective/data formulation rather than
repeat oversampling: add verified negative examples with an explicit `none` target, report per-class
precision/recall/F1, and use a VLM-compatible completion-only collator if assistant masking is needed.

## Completion-only iteration

The third iteration changed AGDD into TRL's conversational prompt-completion format and enabled
`completion_only_loss`. This is supported for VLM prompt-completion data even though
`assistant_only_loss` is not. It excludes prompt and image tokens from the loss and trains only on
the expected defect list. No oversampling was used. Checkpoint 40 was selected on held-out loss.

| Model | Exact match | Macro-F1 | Micro-F1 | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen2.5-VL-3B | 4.55% (1/22) | 0.222 | 0.274 | 606 ms |
| Completion-only QLoRA | **22.73% (5/22)** | **0.494** | **0.636** | 806 ms |

| Class | Base F1 | Tuned precision | Tuned recall | Tuned F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| contusion | 0.000 | 0.550 | 0.917 | 0.687 | 12 |
| scratches | 0.000 | 0.583 | 0.467 | 0.519 | 15 |
| crack | 0.250 | 0.000 | 0.000 | 0.000 | 3 |
| spot | 0.636 | 0.909 | 0.667 | 0.769 | 15 |

This adapter is the first promoted tuning candidate because it improves exact match by 18.18
percentage points, macro-F1 by 0.272, and micro-F1 by 0.362 on the same 22 protected examples.
It is not a final model: latency increases by 200 ms and crack recall regresses from 100% to 0%.
The next iteration should target the three crack examples through loss weighting or carefully bounded
sampling while retaining completion-only loss, and must reject any candidate that loses the aggregate
gains above.

## Crack-targeted completion-only iteration

A fourth run combined completion-only loss with the existing 7x repeat factor for training examples
containing crack. It recovered crack recall but failed the predeclared promotion gate because protected
exact match regressed.

| Model | Exact match | Macro-F1 | Micro-F1 | Crack F1 | Mean latency | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Completion-only | **22.73% (5/22)** | 0.494 | 0.636 | 0.000 | **806 ms** | Promote |
| Completion-only + crack repeat | 9.09% (2/22) | **0.607** | **0.647** | **0.353** | 934 ms | Reject |

The targeted adapter reached crack precision 0.214 and recall 1.0, but it predicted crack on 14/22
examples and reduced exact-set correctness. This demonstrates a real tradeoff rather than a universal
improvement. The untargeted completion-only adapter remains the flagship checkpoint; future work
should tune a class-specific threshold or use a softer weighted loss rather than 7x duplication.

## Frozen protected-split rerun

The earlier 22-example figures above are validation diagnostics because the same official AGDD
validation set selected the checkpoint. To remove that leakage, the official validation IDs were
frozen before retraining into 11 selection examples and 11 protected-test examples. The exact
manifest is `data/splits/agdd_val_selection_test_v1.json` (SHA-256
`bc530a6b7b99142dc6bb71aa44b3fbad56c175afe537303c5a919992d1cb6b04`). Checkpoint 40 was selected
only from selection loss (best loss 0.4422). Each model was then evaluated on the protected subset
once on a Colab Tesla T4.

| Model | Exact match (95% bootstrap CI) | Macro-F1 | Micro-F1 | ECE | Brier | Mean latency (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 9.09% (0–27.27%) | 0.177 | 0.229 | 0.638 | 0.499 | 707 ms (542–989) |
| Completion-only QLoRA | **27.27% (0–54.55%)** | **0.494** | **0.636** | **0.459** | **0.344** | 924 ms (742–1,219) |
| SmolVLM2-2.2B base | 9.09% (0–27.27%) | 0.361 | 0.412 | 0.462 | **0.317** | 1,319 ms (1,175–1,587) |

The tuned checkpoint improves exact set match by 18.18 percentage points and improves both aggregate
F1 and calibration, but costs 218 ms mean latency. The sample is deliberately small, so the exact
match confidence intervals overlap and this is a promising pilot rather than a definitive superiority
claim. Crack remains the clearest regression: base crack recall was 1.0 (low precision), while tuned
crack recall was 0.0. Tuned contusion recall reached 1.0, scratches recall 0.5, and spot recall 0.625.
The compact evidence record is `reports/agdd_protected_qwen25_vl_3b_summary.json`; it includes raw
report hashes and the bootstrap configuration.

SmolVLM2 is an independent model family, not another Qwen checkpoint. It tied base Qwen on exact
set match, exceeded it on macro/micro-F1 and calibration, and was the slowest model. Its principal
failure was contusion recall 0.0; crack precision/recall were 0.333/0.5, scratches 0.5/0.5, and spot
1.0/0.375. Model choice and prompt were fixed before its one-time protected run.
