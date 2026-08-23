# Run the development VLM batch on a free Kaggle GPU

The Transformers backend removes the Apple-Silicon-only execution constraint.
It loads Qwen2-VL once in 4-bit mode, reuses that CUDA session across the fixed
12-case development queue, validates every response against the request-bound
schema, and writes a provenance record outside Git.

## Kaggle workflow

1. Create a Kaggle notebook, enable Internet, and choose the P100 accelerator.
2. Import `notebooks/kaggle_vlm_batch.ipynb` and run every cell.
3. Confirm the dry run reports 12 development cases and no inference.
4. Run the real batch. Download the JSON record and the archive whose SHA-256
   is printed by the final cell.
5. Inspect failures and rejected-output hashes. Do not silently repair invalid
   responses or treat execution success as evaluator accuracy.

The notebook installs the PyTorch 2.7.1 CUDA 12.6 wheel before the project
dependencies. Kaggle's newer default wheel omits Pascal `sm_60` kernels and
otherwise fails during model loading on a Tesla P100 with `no kernel image is
available for execution on the device`.

The run closes the Linux/CUDA execution gap and provides real model outputs. It
does not establish human alignment. Accuracy claims still require two genuine
independent raters, adjudication, and comparison against the frozen reference.

## First measured P100 batch

The first completed free-GPU batch attempted all 12 fixed development cases
with one reused Qwen2-VL session. Eight responses passed the request-bound
schema and four were retained as JSON-parse failures, for an execution success
rate of 66.7%. The result supports the execution path and exposes a remaining
structured-generation limitation; it does not support an evaluator-quality or
human-alignment claim.

The compact run record, exact code commit, environment, failure counts, full
archive digest, and Kaggle notebook link are stored in
[`reports/kaggle_vlm_batch_v0_1.json`](../reports/kaggle_vlm_batch_v0_1.json).
