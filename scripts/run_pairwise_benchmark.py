from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.pairwise_judge import (
    JudgeOrientation,
    create_mlx_pairwise_session,
    judge_example,
)
from aerosynth_eval.preference_benchmark import BenchmarkPartition, load_manifest

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--model", default="mlx-community/Qwen2-VL-2B-Instruct-4bit")
parser.add_argument("--partition", choices=[p.value for p in BenchmarkPartition], default="heldout")
parser.add_argument("--limit", type=int, default=10)
parser.add_argument("--swap", action="store_true")
parser.add_argument(
    "--output", type=Path, default=Path("outputs/preference_judgments/judgments.jsonl")
)
args = parser.parse_args()
examples = [x for x in load_manifest(args.manifest) if x.partition.value == args.partition][
    : args.limit
]
infer = create_mlx_pairwise_session(args.model)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as stream:
    for example in examples:
        orientations = (
            [JudgeOrientation.STANDARD, JudgeOrientation.SWAPPED]
            if args.swap
            else [JudgeOrientation.STANDARD]
        )
        for orientation in orientations:
            record = judge_example(
                example, model_id=args.model, inference=infer, orientation=orientation
            )
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
print(args.output)
