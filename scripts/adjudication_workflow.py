from __future__ import annotations

import argparse
from pathlib import Path

from aerosynth_eval.adjudication import prepare_adjudication_queue, validate_completed_adjudication

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
p = sub.add_parser("prepare")
p.add_argument("rater_a", type=Path)
p.add_argument("rater_b", type=Path)
p.add_argument("output", type=Path)
v = sub.add_parser("validate")
v.add_argument("path", type=Path)
args = parser.parse_args()
print(
    prepare_adjudication_queue(args.rater_a, args.rater_b, args.output)
    if args.command == "prepare"
    else validate_completed_adjudication(args.path)
)
