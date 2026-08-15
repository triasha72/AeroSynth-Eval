"""Executor-neutral shard planning and deterministic result reduction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.scalable_evaluation import shard_index


class ShardPlan(BaseModel):
    """Deterministic assignment of examples to one execution shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_shards: int = Field(ge=1)
    shard_index: int = Field(ge=0)
    example_ids: list[str]


def build_shard_plans(
    example_ids: list[str],
    num_shards: int,
) -> list[ShardPlan]:
    """Build deterministic, non-overlapping shard plans."""

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")

    counts = Counter(example_ids)

    duplicates = sorted(example_id for example_id, count in counts.items() if count > 1)

    if duplicates:
        raise ValueError("Duplicate example IDs are not allowed: " + ", ".join(duplicates))

    buckets: list[list[str]] = [[] for _ in range(num_shards)]

    for example_id in sorted(example_ids):
        index = shard_index(
            example_id,
            num_shards,
        )
        buckets[index].append(example_id)

    return [
        ShardPlan(
            num_shards=num_shards,
            shard_index=index,
            example_ids=sorted(ids),
        )
        for index, ids in enumerate(buckets)
    ]


def reduce_shard_jsonl(
    paths: list[Path],
    output_path: Path,
    expected_ids: set[str],
) -> Path:
    """Reduce shard outputs with strict completeness checks."""

    records: dict[str, dict[str, object]] = {}

    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(
                stream,
                start=1,
            ):
                if not line.strip():
                    continue

                raw_payload: object = json.loads(line)

                if not isinstance(raw_payload, dict):
                    raise ValueError(
                        f"Distributed result must be a JSON object: {path}:{line_number}"
                    )

                payload: dict[str, object] = {str(key): value for key, value in raw_payload.items()}

                if "example_id" not in payload:
                    raise ValueError(
                        f"Distributed result is missing example_id: {path}:{line_number}"
                    )

                example_id = str(payload["example_id"])

                if example_id in records:
                    raise ValueError(f"Duplicate distributed result '{example_id}'.")

                records[example_id] = payload

    observed_ids = set(records)

    missing = expected_ids - observed_ids
    extra = observed_ids - expected_ids

    if missing or extra:
        raise ValueError(
            "Shard reduction completeness failure: "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for example_id in sorted(records):
            stream.write(
                json.dumps(
                    records[example_id],
                    sort_keys=True,
                )
                + "\n"
            )

    return output_path
