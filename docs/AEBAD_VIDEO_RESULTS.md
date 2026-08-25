# Native AeBAD-V temporal benchmark

Date: 2026-08-25

This benchmark uses the official CC BY 4.0 AeBAD-V aero-engine blade inspection sequences, not
derived still-image clips. The 1.62 GB official archive contained 3,410 ordered JPEG frames. Eight
balanced windows from `video1` were used for interface selection. Sixteen balanced windows from the
unseen `video2` and `video3` viewpoints were frozen as the protected test. Every window contains 16
checksummed source frames; the deterministic manifest generator is
`scripts/freeze_aebad_video_split.py`.

The free-form interface failed selection through description generation, and tiled 16-frame input
exceeded a 15 GB T4. Before opening the protected videos, the deployment interface was frozen to
384-pixel non-tiled frames and constrained next-token scoring of the single-token labels ` good` and
` anomaly`. This selection history is reported rather than hidden.

| Frames | Accuracy | Balanced accuracy | Anomaly recall | ECE | Mean latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50% | 50% | 0% | 0.495 | 168 ms |
| 4 | 50% | 50% | 0% | 0.489 | 251 ms |
| 8 | 50% | 50% | 0% | 0.490 | 536 ms |
| 16 | 50% | 50% | 0% | 0.478 | 1,070 ms |

Both protected viewpoint slices scored 50% balanced accuracy. The model predicted `GOOD` for every
protected window. Multi-frame context therefore produced no quality gain while 16-frame latency was
6.4 times one-frame latency. This closes the native-video execution gap, but it does **not** close the
temporal-reasoning quality gap. The next credible experiment is AeBAD-V adaptation on training/video1
only, followed by one new test on a separately frozen set; repeated prompt tuning on videos 2–3 is
prohibited.

Machine-readable evidence and raw-output hashes are stored in
`reports/aebad_video_smolvlm2_summary.json`.
