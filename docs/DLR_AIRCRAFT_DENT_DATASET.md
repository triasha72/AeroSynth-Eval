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

Then download the archive using the URL recorded in that receipt, extract it
outside Git, and audit its directory layout, image count, annotation count,
duplicate images, and group split before training. No train/test split is
assumed from the Zenodo metadata alone.

This dataset can address the 200-image real-test-size gap. It does not yet close
the gap: a completed experiment still needs five evaluator-selected subsets,
an untouched 200+ image test set, and a leakage check.

The dent labels are derived from optical tracking. They are useful for a real
inspection-image benchmark, but they are not independent human ratings of
AeroSynth-generated images.
