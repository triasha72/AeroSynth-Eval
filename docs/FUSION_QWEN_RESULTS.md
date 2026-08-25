# Transcript fusion robustness

Date: 2026-08-25

Qwen2.5-VL-3B was evaluated on all 12 AeroSynth region-condition cases under five modality
conditions. Visual modes used four ordered derived frames. Transcript modes used deterministic
condition-grounded operator-note templates.

| Mode | Accuracy | Latency p50 | Latency p95 |
| --- | ---: | ---: | ---: |
| Vision only | 16.7% | 451 ms | 821 ms |
| Transcript only | 100% | 163 ms | 216 ms |
| Frames + transcript | 75% | 486 ms | 513 ms |
| Missing transcript | 16.7% | 386 ms | 475 ms |
| Frames + noisy transcript | 75% | 494 ms | 532 ms |

The system is robust to the deterministic corruption pattern but not to a missing transcript. Fusion
underperforms transcript-only, showing that weak visual evidence can override an otherwise informative
note. This is a useful missing-modality and conflict result.

The transcripts directly encode condition semantics and are synthetic templates. Therefore 100%
transcript accuracy is an expected upper bound, not proof of speech recognition or natural spoken
language understanding. No waveform or ASR model was evaluated. A publishable audio claim still
requires independently collected or public audio/video clips, real transcripts, and an ASR corruption
evaluation.
