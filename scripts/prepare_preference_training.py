from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from aerosynth_eval.preference_benchmark import load_manifest
from aerosynth_eval.preference_training import (
    AdaptationConfig,
    prepare_mlx_vlm_dataset,
    training_command,
)

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument(
    "--dataset-out", type=Path, default=Path("data/external/training/genai_preference")
)
parser.add_argument(
    "--adapter-out", type=Path, default=Path("outputs/adapters/genai_preference.safetensors")
)
args = parser.parse_args()
path = prepare_mlx_vlm_dataset(load_manifest(args.manifest), args.dataset_out)
config = AdaptationConfig(learning_rate=2e-5, batch_size=1, epochs=1, lora_rank=8, lora_alpha=16)
print(
    json.dumps(
        {
            "dataset": str(path),
            "training_command": shlex.join(training_command(config, path, args.adapter_out)),
        },
        indent=2,
    )
)
