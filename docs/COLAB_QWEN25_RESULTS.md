# Qwen2.5-VL-3B Colab development result

Date: 2026-08-25

The public `Qwen/Qwen2.5-VL-3B-Instruct` model was executed on the frozen 12-case development
queue using a free Google Colab Tesla T4 (15,360 MiB). The protected test set was not touched.

| Metric | Result |
| --- | ---: |
| Cases | 12 |
| Parse rate | 100% |
| Accuracy | 33.3% |
| Mean generation latency | 1,246 ms/case |

The model predicted `no_visible_defect` for 11 of 12 cases and `corrosion` once. It correctly
classified all three clean cases and one corrosion case, but missed every crack and coating-damage
case. This is a genuine negative accuracy result and a useful failure slice: the stronger model and
GPU dramatically improved execution speed and output-format reliability, but did not improve
aggregate accuracy over the prompted SmolVLM-256M CPU baseline (also 33.3%).

The exact report is `reports/qwen25_vl_3b_development.json`, with SHA-256
`420553c19d7a716135f58893d4897e3e73a915b924afb8d3f6cdc4ecef8bb1c4`.

Do not promote this prompt to the protected test set. The next experiment should use the same model
with a defect-sensitive prompt and/or few-shot examples, still on development data. Freeze that
prompt only after it improves defect recall without losing clean-case performance.
