from __future__ import annotations

import argparse
from pathlib import Path

from aerosynth_eval.experiment_hillclimbing import ExperimentResult, rank_experiments

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()
rows = [
    ExperimentResult.model_validate_json(x)
    for x in args.results.read_text().splitlines()
    if x.strip()
]
[print(x.model_dump_json()) for x in rank_experiments(rows)]
