# Free stronger-model runbook

The selected stronger baseline is `Qwen/Qwen2.5-VL-3B-Instruct`. Its weights are public, but a
GPU is required for a practical run. A free Google Colab T4 is the primary path; Kaggle's free GPU
notebook is the fallback. The protected test set remains untouched: first run the 12-case
development queue, freeze the prompt, then run the protected set once.

## Colab or Kaggle

Upload the `AeroSynth-Eval` directory (or mount Drive), enable a GPU runtime, open a terminal cell,
and run:

```bash
%cd /content/AeroSynth-Eval
!pip install -q "transformers>=4.50,<5" accelerate qwen-vl-utils torch torchvision
!python scripts/run_qwen25_vl.py \
  --output reports/qwen25_vl_3b_development.json
```

Start with `--limit 1` if the runtime has less than 12 GB of available GPU memory. If model loading
runs out of memory, use the free fallback `HuggingFaceTB/SmolVLM-500M-Instruct` rather than altering
the benchmark labels. Do not report a Qwen result until the JSON report exists.

## Acceptance gate

Compare Qwen against `reports/cpu_vlm_prompted_development12.json` using the same 12 cases. Promote
it only if it improves accuracy without reducing parse rate. Report class slices and regressions,
not only the aggregate. The current reproducible floor is 33.3% accuracy and 91.7% parse rate from
SmolVLM-256M-Instruct.

Free hosted GPU quotas and availability are not guaranteed. A local NVIDIA GPU, Apple Silicon with
MLX, or a paid GPU are execution alternatives; they do not change the evaluation protocol.
