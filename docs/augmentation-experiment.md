# Frozen downstream augmentation experiment

`run_downstream_utility.py` compares real-only training, equal-budget random
augmentation and evaluator-selected augmentation over matched seeds. A small
pixel/logistic-regression classifier is the declared baseline, not a defect detector.
Validation selects its decision threshold; real test images evaluate every frozen arm.

```sh
pip install -e '.[real-data]'
python scripts/run_downstream_utility.py --manifest manifest.json --data-root data --budget 100 --seeds 17 29 41 53 67 --output results/utility-v1.json
```

The manifest is a JSON list. Every record has `id`, relative `path`, `group`,
`split` (train/val/test), `kind` (real/synthetic), and binary `label`.
Synthetic records must be train-only and add `evaluator_score`; where generated
from a source image, set `source_group` to its real training group. Never omit
known ancestry. Both classes must occur in each real split. The loader rejects
cross-split real groups, duplicate image bytes and paths outside the data root.

Freeze evaluator training and scores without consulting real test labels. Use
actual acquisition/source groups, not one invented group per correlated image.
The receipt includes image hashes, seed results, test F1 and defect recall. Its
criteria require 200 test images and five seeds, improvement over both controls,
and no more than 0.05 recall loss. These are descriptive criteria, not a significance
test. A real audited manifest, synthetic image set and uncertainty study are still
needed before claiming useful augmentation. Generated test fixtures verify code only.
