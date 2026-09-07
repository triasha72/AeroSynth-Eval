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
the released images. Use it to map annotations to images, check duplicates, and
define a group-aware train/validation/test split. No split is assumed from the
Zenodo metadata alone.

This dataset can address the 200-image real-test-size gap. It does not yet close
the gap: a completed experiment still needs five evaluator-selected subsets,
an untouched 200+ image test set, and a leakage check.

The dent labels are derived from optical tracking. They are useful for a real
inspection-image benchmark, but they are not independent human ratings of
AeroSynth-generated images.
