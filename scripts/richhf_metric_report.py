from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.richhf_evaluation import dimension_metric

parser = argparse.ArgumentParser()
parser.add_argument("pairs", type=Path, help="JSONL rows with human and predicted numeric values")
args = parser.parse_args()
rows = [json.loads(x) for x in args.pairs.read_text().splitlines() if x.strip()]
print(
    dimension_metric(
        [float(x["human"]) for x in rows], [float(x["predicted"]) for x in rows]
    ).model_dump_json(indent=2)
)
