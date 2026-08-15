from __future__ import annotations

import argparse
from pathlib import Path

from aerosynth_eval.human_agreement import summarize_human_agreement

parser = argparse.ArgumentParser()
parser.add_argument("rater_a", type=Path)
parser.add_argument("rater_b", type=Path)
args = parser.parse_args()
print(summarize_human_agreement(args.rater_a, args.rater_b).model_dump_json(indent=2))
