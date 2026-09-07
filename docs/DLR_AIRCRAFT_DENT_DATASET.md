# DLR aircraft-dent dataset

This is the second real-aircraft dataset track for AeroSynth-Eval. The public
Zenodo release was produced with DLR's aircraft maintenance research group. It
reports more than 6,000 labelled dent images and is released under MIT.

The archive is 5.7 GB. The repository does not download or commit it during CI.
First create a provenance receipt:

```bash
PYTHONPATH=src python scripts/audit_dlr_dent_release.py \
  --output reports/dlr_aircraft_dent_release_v1.json
```

Then download the archive using the URL recorded in that receipt. The command
is resumable and asks for an explicit confirmation because the archive is
large. It checks that at least 12 GiB of local space is available.

```bash
PYTHONPATH=src python scripts/download_dlr_dent_archive.py \
  --url "https://zenodo.org/api/records/17900121/files/plane-10.27-fulldata-split.zip/content" \
  --confirm-large-download
PYTHONPATH=src python scripts/inspect_dlr_dent_archive.py \
  --archive data/external/dlr-aircraft-dent/plane-10.27-fulldata-split.zip \
  --output reports/dlr_aircraft_dent_archive_inventory_v1.json
```

The inventory reads ZIP metadata only. It records directory layout, image and
annotation member counts, and safe sample paths without extracting or committing
the released images.

The release contains `train`, `val`, and `test` folders, but they should not be
used as the final score split. Capture timestamps from nearby sessions appear in
more than one released folder. The baseline below rebuilds train, validation,
and test partitions at a 300-second capture-session boundary and keeps the test
records unseen during threshold selection.

```bash
PYTHONPATH=src python scripts/train_dlr_dent_baseline.py \
  --archive data/external/dlr-aircraft-dent/plane-10.27-fulldata-split.zip \
  --output reports/dlr_aircraft_dent_baseline_v1.json
```

It reads the archive directly and writes a small, source-free receipt with the
real-data triage result. The first model predicts whether an inspection image
contains a released dent label. It is deliberately a simple baseline; it does
not claim to localize dents or approve maintenance work.

## Measured first baseline

The initial run used 2,087 training images, 492 validation images, and 645
session-held-out test images. It selected a high-recall threshold on validation
and reached test recall of `0.9777`, precision of `0.4340`, F1 of `0.6011`, and
ROC-AUC of `0.5969`. The false-positive count was high (`343` of `376` negative
test images), so this model is a starting point for data and model work, not a
useful inspection decision tool. The exact receipt is
`reports/dlr_aircraft_dent_baseline_v1.json`.

This dataset can address the 200-image real-test-size gap. It does not provide
five evaluator-selected subsets or a human review of AeroSynth-generated
images, so those remain separate evidence gaps.

The dent labels are derived from optical tracking. They are useful for a real
inspection-image benchmark, but they are not independent human ratings of
AeroSynth-generated images.
