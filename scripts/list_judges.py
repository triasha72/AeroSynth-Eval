from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.judge_registry import load_judge_specs

parser = argparse.ArgumentParser()
parser.add_argument("registry", type=Path)
args = parser.parse_args()
print(json.dumps([x.model_dump(mode="json") for x in load_judge_specs(args.registry)], indent=2))
