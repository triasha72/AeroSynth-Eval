from __future__ import annotations

import argparse
from pathlib import Path

from aerosynth_eval.human_alignment import (
    EvaluatorCase,
    HumanReferenceCase,
    summarize_human_alignment,
)

parser = argparse.ArgumentParser()
parser.add_argument("human", type=Path)
parser.add_argument("evaluator", type=Path)
args = parser.parse_args()
human = [
    HumanReferenceCase.model_validate_json(x)
    for x in args.human.read_text().splitlines()
    if x.strip()
]
evaluator = [
    EvaluatorCase.model_validate_json(x)
    for x in args.evaluator.read_text().splitlines()
    if x.strip()
]
print(summarize_human_alignment(human, evaluator).model_dump_json(indent=2))
