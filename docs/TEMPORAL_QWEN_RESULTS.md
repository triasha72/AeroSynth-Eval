# Temporal Qwen2.5-VL-3B ablation

Date: 2026-08-25

Qwen2.5-VL-3B was evaluated on all 12 AeroSynth region-condition cases with 1, 4, 8,
and 16 ordered frames. Each sequence was deterministically derived from the four independently
rendered capture profiles for a case using small camera-motion and brightness variations.

This is a derived temporal stress test, not real captured aircraft video. Its purpose is to measure
whether added ordered visual context changes model quality and systems cost under a controlled label.

| Frames | Cases | Accuracy | Latency p50 | Latency p95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 25% | 182 ms | 208 ms |
| 4 | 12 | 25% | 402 ms | 438 ms |
| 8 | 12 | 25% | 711 ms | 759 ms |
| 16 | 12 | 25% | 1,491 ms | 1,524 ms |

The model predicted `clean` for every case at every frame count. Consequently, each setting correctly
classified the three clean cases and missed all coating, corrosion, and crack cases. Multi-frame
context provided no quality gain, while 16-frame p50 latency was 8.2x the one-frame latency.

This rejects the hypothesis that simply adding frames improves the current system. The next temporal
iteration needs real or independently labeled inspection clips and a model/prompt adapted for the
four AeroSynth conditions. The current result is still useful deployment evidence because it bounds
the compute cost and demonstrates an executed 1/4/8/16-frame evaluation path.
