# AGDD real aircraft-image baseline

This track evaluates real aircraft-domain imagery without mixing it into
AeroSynth-Eval's procedural protected test set. AGDD contains paired forward-
and backward-illumination images of aircraft glass-canopy defects with oriented
and rectangular detection annotations.

## Provenance and scope

- Source: `https://github.com/core128/AGDD`
- Pinned commit: `4b5daa92929934f30b1155033c3ce67b7701960f`
- License: CC BY-NC-SA 4.0
- Train split: 197 paired samples
- Validation split: 22 paired samples
- Classes: contusion, scratches, crack, and spot

The audit validates paired filenames, PNG dimensions, class identifiers,
normalized annotation coordinates, and agreement between oriented and
rectangular label files. Images are not redistributed in this repository.

## First measured baseline

The transparent baseline downsamples both illumination views to 32x32
grayscale pixels and trains four class-balanced one-vs-rest logistic models.
The official validation split is never used for fitting.

| Validation metric | Result |
|---|---:|
| Macro F1 | 0.6108 |
| Micro F1 | 0.5926 |
| Exact multilabel match | 0.1818 |
| Contusion F1 | 0.6667 |
| Scratches F1 | 0.5714 |
| Crack F1 | 0.6667 |
| Spot F1 | 0.5385 |

Only three validation samples contain a crack label, so the corresponding F1
estimate is highly uncertain. These results are a real-domain construction
baseline, not evidence of maintenance suitability, defect-size estimation, or
airworthiness decision quality.

## Reproduction

```bash
python -m pip install -e '.[real-data]'
python scripts/audit_agdd.py /path/to/pinned/AGDD reports/agdd_audit.json
python scripts/train_agdd_real_baseline.py \
  /path/to/pinned/AGDD reports/agdd_real_baseline_v1.json
```

Commercial use requires separate licensing review because AGDD is
noncommercial and share-alike.

## Operational release decision

The tracked operational policy rejects this baseline. AGDD does not permit
commercial use, the validation split has 22 pairs rather than the required 200,
and macro F1 is below the frozen 0.80 threshold. Exact-match accuracy is 4/22;
its Wilson 95% interval is recorded to expose the uncertainty hidden by the
point estimate. CI regenerates `reports/agdd_release_assessment_v1.json` and
will fail if the evidence and release decision diverge.

## Procedural-to-real transfer result

A controlled 10-seed low-data experiment compares equal-sized crack classifiers
trained on real AGDD images, procedural images, or a 50/50 mixture. Every
treatment is evaluated on the same untouched 44 real AGDD validation images
(22 paired cases, only six crack-positive images).

| Training treatment | Macro F1 mean ± SD | Crack recall mean ± SD |
|---|---:|---:|
| Real only | 0.3881 ± 0.0977 | 0.4000 ± 0.2854 |
| Procedural-only control | 0.2567 ± 0.1765 | 0.6000 ± 0.5164 |
| Real + procedural | 0.4419 ± 0.0780 | 0.3167 ± 0.2144 |

The mixture improved overall binary macro F1 but reduced crack recall. The
procedural-only control was highly unstable. Therefore this experiment does not
support a claim that synthetic augmentation improves safety-relevant crack
detection. It demonstrates the intended synthetic-to-real measurement boundary
and identifies the need for a larger real protected test set. Full per-seed
evidence is in `reports/agdd_transfer_experiment_v1.json`.
