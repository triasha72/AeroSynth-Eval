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

The run closes the Linux/CUDA execution gap and provides real model outputs. It
does not establish human alignment. Accuracy claims still require two genuine
independent raters, adjudication, and comparison against the frozen reference.
