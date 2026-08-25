# Streaming Qwen2.5-VL-3B replay

Date: 2026-08-25

The temporal inference path was executed as a recorded-frame replay with a bounded drop-oldest input
queue of capacity 2 and a sliding visual window of up to 4 frames. Three operating modes exercise
normal processing, overload, and an explicitly missing input.

| Mode | Windows | Accuracy | p50 | p95 | Throughput | Drops | Recovered cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 48 | 16.7% | 342 ms | 682 ms | 2.50 windows/s | 0 | n/a |
| Overload | 24 | 25% | 273 ms | 351 ms | 3.47 windows/s | 24 | 12/12 |
| Injected drop | 24 | 25% | 281 ms | 384 ms | 3.17 windows/s | 24 | 12/12 |

Peak CUDA memory was 7,241 MiB allocated and 7,318 MiB reserved on a Tesla T4. The pipeline
continued producing outputs after every overloaded or injected-drop case. Higher throughput in the
drop modes is not a quality improvement: fewer and shorter windows were processed.

This establishes bounded queues, sliding windows, latency p50/p95, throughput, GPU memory, and
post-drop recovery on a real VLM. It is a recorded replay, not a live camera or concurrent production
service. Accuracy remains weak and is consistent with the temporal ablation's visual failure mode.
