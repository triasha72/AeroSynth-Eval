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
