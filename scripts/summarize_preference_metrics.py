from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerosynth_eval.pairwise_judge import PairwiseJudgmentRecord
from aerosynth_eval.preference_metrics import summarize_preference_metrics

parser = argparse.ArgumentParser()
parser.add_argument("judgments", type=Path)
args = parser.parse_args()
records = [
    PairwiseJudgmentRecord.model_validate_json(line)
    for line in args.judgments.read_text().splitlines()
    if line.strip()
]
print(
    json.dumps(
        summarize_preference_metrics(records).model_dump(mode="json"), indent=2, sort_keys=True
    )
)
