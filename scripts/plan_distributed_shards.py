from __future__ import annotations

import argparse
import json

from aerosynth_eval.distributed_execution import build_shard_plans

parser = argparse.ArgumentParser()
parser.add_argument("--num-shards", type=int, default=4)
parser.add_argument("example_ids", nargs="+")
args = parser.parse_args()
print(
    json.dumps(
        [x.model_dump(mode="json") for x in build_shard_plans(args.example_ids, args.num_shards)],
        indent=2,
    )
)
