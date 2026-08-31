# AeroSynth-Eval: the next useful steps

[Read the project overview and measured results](README.md)

The repository can now run a controlled comparison of real-only,
synthetic-only, and mixed training against untouched public AGDD images. Ten
matched seeds showed a genuine trade-off: mixed training raised mean macro F1
from `0.3881` to `0.4419`, while mean crack recall fell from `0.4000` to
`0.3167`.

That result changes the plan. The next goal is not to generate more images. It
is to understand why augmentation helps the average while hurting the defect
that matters most.

## Real-image work

- Choose augmentation settings on development data with crack recall protected,
  then evaluate once on the held-out AGDD pairs.
- Inspect errors by lighting pair and condition rather than relying on one
  average score.
- Find a second public aircraft inspection dataset with a compatible license and
  label space. Cross-dataset testing is needed before claiming transfer.

## Human and VLM work

Two people still need to rate the blinded image set independently. Their
agreement and adjudicated labels will provide the reference needed to evaluate
the visual-language model. Only then does it make sense to report calibration or
human alignment.

The execution tools for local and Kaggle VLM batches are already present, and
failed responses are retained. They prove that the pipeline runs; they do not
prove evaluator quality.

## Boundary that will remain

This is a dataset-quality research tool, not an airworthiness, maintenance
release, or defect-diagnosis system. A model change must improve performance on
real held-out images without sacrificing a protected defect-recall measure.
