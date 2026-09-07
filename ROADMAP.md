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

- [x] add a no-budget volunteer recruitment guide and reproducible 12-case pilot
- [ ] collect two independent pilot submissions from real volunteers
- [x] implement real-test downstream utility as an alternative evidence path
- [ ] run five frozen selection experiments on a 200+ image real test set

The public GenAI-Bench image-generation split is now materialized: 1,735 real
human preference votes, with prompt-grouped train, validation, and held-out
partitions and a checksummed text-free artifact. The next compute step is to
run the frozen VLM judge on the 235 held-out pairs and report agreement, order
bias, invalid-response rate, latency, and cost without tuning on that partition.

Two people still need to rate the blinded image set independently. Their
agreement and adjudicated labels will provide the reference needed to evaluate
the visual-language model. Only then does it make sense to report calibration or
human alignment.

The execution tools for local and Kaggle VLM batches are already present, and
failed responses are retained. They prove that the pipeline runs; they do not
prove evaluator quality.

The combined transfer gate is now executable. It requires a larger real test
set, a macro-F1 improvement, no material crack-recall regression, and genuine
independent human alignment. The current mixed model is blocked on the real
sample-size and crack-recall checks.

## Boundary that will remain

This is a dataset-quality research tool, not an airworthiness, maintenance
release, or defect-diagnosis system. A model change must improve performance on
real held-out images without sacrificing a protected defect-recall measure.
