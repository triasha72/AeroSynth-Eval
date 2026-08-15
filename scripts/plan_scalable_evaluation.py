from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from aerosynth_eval.scalable_evaluation import EvaluationRunConfig, build_experiment_manifest

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--model", required=True)
parser.add_argument("--num-shards", type=int, default=1)
args = parser.parse_args()
sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
config = EvaluationRunConfig(
    dataset_id="GenAI-Bench",
    dataset_revision="main",
    dataset_sha256=sha,
    model_id=args.model,
    model_revision="main",
    prompt_version="pairwise-v0.1",
    max_tokens=500,
    temperature=0.0,
    num_shards=args.num_shards,
)
print(
    json.dumps(
        build_experiment_manifest(config, git).model_dump(mode="json"),
        indent=2,
        default=str,
        sort_keys=True,
    )
)
