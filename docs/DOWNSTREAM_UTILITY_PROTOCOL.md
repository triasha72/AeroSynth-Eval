# Downstream utility protocol

This experiment tests a practical claim: do synthetic images rated more highly
by the evaluator produce a better detector on untouched real aircraft images?

Create at least five synthetic training subsets at different evaluator-score
levels. Keep model architecture, optimization, real training rows, and random
seeds fixed. The real test set cannot be used to choose images, prompts,
thresholds, or checkpoints.

Each experiment row records `mean_evaluator_score` and `real_macro_f1_gain`
against the same real-only baseline. The gate requires:

- at least five selection experiments;
- at least 200 untouched real test images;
- no real-test leakage;
- Spearman correlation of at least 0.5 between evaluator score and real F1 gain;
- positive mean real-image F1 gain.

```json
{
  "untouched_real_test_images": 240,
  "real_test_used_for_selection": false,
  "selection_experiments": [
    {"mean_evaluator_score": 0.41, "real_macro_f1_gain": 0.003}
  ]
}
```

Run the check with:

```bash
PYTHONPATH=src python scripts/assess_downstream_utility.py \
  --experiment outputs/downstream_selection_experiment.json \
  --output reports/downstream_utility_assessment_v1.json
```

AGDD currently has only 44 protected real images, so the published result cannot
pass the 200-image requirement. A second licensed real aircraft dataset is still
needed. Dataset labels count as public human-labelled evidence; they are not new
independent reviews of AeroSynth's generated images.
