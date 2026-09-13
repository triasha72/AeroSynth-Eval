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

## Paired uncertainty

The runner now retains per-seed test predictions and test IDs and computes paired
95% acquisition-group bootstrap intervals for selected-minus-real and
selected-minus-random F1/recall. The same sampled groups are used for each arm and
seed; seed metrics are averaged within each draw. One-class draws are excluded and
the valid-draw fraction is reported. Fewer than 90% usable draws flags an unstable
estimate. These intervals condition on the supplied groups and fitted models, and
do not establish independence of the source data or account for all training-set
uncertainty. The descriptive decision gate remains separate from these intervals.

## Public DLR source

Zenodo record `17900121` lists the aircraft-dent data under an MIT license. The
two archives are large: `plane-10.27-fulldata-split.zip` is approximately 5.74 GB
and `plates-11.08-fulldata-split.zip` is approximately 3.84 GB. Inspect the record
and confirm available disk space before downloading. The repository downloader
requires an explicit large-download flag:

```sh
python scripts/download_dlr_dent_archive.py \
  --url https://zenodo.org/api/records/17900121/files/plane-10.27-fulldata-split.zip/content \
  --confirm-large-download --minimum-free-gb 12
```

Keep the archive outside Git and retain the Zenodo record ID, SHA-256, license,
split files and annotation documentation with the experiment receipt. This is a
dent dataset; it does not automatically satisfy a broader aircraft-defect claim.
Use its actual split and group identifiers when constructing the utility manifest.
