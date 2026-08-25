# AGDD aircraft-domain extension

AGDD is now pinned as the aircraft-domain extension for AeroSynth-Eval. It contains 219
aircraft glass-canopy samples. Every sample has two distinct 640×640 views captured under
forward and backward illumination, plus matching oriented and rectangular bounding-box
annotations. The official train/validation split is retained.

## Verified inventory

| Split | Pairs | Images | Contusion | Scratches | Crack | Spot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 197 | 394 | 125 | 189 | 25 | 184 |
| Validation | 22 | 44 | 13 | 28 | 4 | 31 |

The audit checks paired filenames, dimensions, normalized coordinates, known class IDs, and
matching OBB/rectangular class counts. Its deterministic manifest digest is
`fa43094cb00b42808c522d321b69c7ca64695a39a49a0a18047282736df096b8`.

Run it again with:

```bash
python scripts/audit_agdd.py data/public/agdd-official reports/agdd_audit.json
```

Chat-style paired-image manifests are materialized in `data/training/agdd_vlm`. The 197-sample
training split can support a small domain-adaptation experiment; the 22-sample validation split
must remain excluded from training.

## Claims this supports—and does not support

AGDD strengthens aircraft-domain and paired-view perception evidence. It does not provide two
independent quality raters, adjudicated labels, video, speech, or a replacement for AeroSynth's
protected test set. Its boxes are ground truth for detection, not human preference judgments.

The license is CC BY-NC-SA 4.0. Derived data and adapters need a license review before commercial
use. The source is pinned to commit `4b5daa92929934f30b1155033c3ce67b7701960f`.
