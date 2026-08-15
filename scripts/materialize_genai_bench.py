from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.preference_benchmark import materialize_genai_bench

parser = argparse.ArgumentParser()
parser.add_argument("--output-root", type=Path, default=Path("data/external/genai_bench"))
parser.add_argument("--limit-per-task", type=int, default=None)
args = parser.parse_args()
path, summary = materialize_genai_bench(args.output_root, limit_per_task=args.limit_per_task)
print(
    json.dumps({"manifest": str(path), **summary.model_dump(mode="json")}, indent=2, sort_keys=True)
)
