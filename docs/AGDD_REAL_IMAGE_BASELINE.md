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
