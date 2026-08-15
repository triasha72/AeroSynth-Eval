import json
from pathlib import Path

import pytest

from aerosynth_eval.distributed_execution import (
    build_shard_plans,
    reduce_shard_jsonl,
)


def test_shards_cover_and_reduce(
    tmp_path: Path,
) -> None:
    ids = [f"e{i}" for i in range(20)]

    plans = build_shard_plans(ids, 4)

    assert len(plans) == 4

    planned_ids = [example_id for plan in plans for example_id in plan.example_ids]

    assert sorted(planned_ids) == sorted(ids)
    assert len(planned_ids) == len(set(planned_ids))

    paths: list[Path] = []

    for index, plan in enumerate(plans):
        path = tmp_path / f"s{index}.jsonl"
        paths.append(path)

        with path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            for example_id in plan.example_ids:
                stream.write(
                    json.dumps(
                        {
                            "example_id": example_id,
                            "ok": True,
                        }
                    )
                    + "\n"
                )

    output = tmp_path / "all.jsonl"

    reduce_shard_jsonl(
        paths,
        output,
        set(ids),
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == len(ids)

    output_ids = [row["example_id"] for row in rows]

    assert output_ids == sorted(ids)


def test_shard_plans_are_deterministic() -> None:
    first = build_shard_plans(
        ["e4", "e1", "e3", "e2"],
        3,
    )

    second = build_shard_plans(
        ["e2", "e3", "e1", "e4"],
        3,
    )

    assert first == second


def test_zero_shards_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        build_shard_plans([], 0)


def test_duplicate_input_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate example IDs",
    ):
        build_shard_plans(
            ["e1", "e2", "e1"],
            2,
        )


def test_duplicate_results_are_rejected(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first.write_text(
        '{"example_id":"e1"}\n',
        encoding="utf-8",
    )

    second.write_text(
        '{"example_id":"e1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate distributed result",
    ):
        reduce_shard_jsonl(
            [first, second],
            tmp_path / "out.jsonl",
            {"e1"},
        )


def test_missing_results_are_rejected(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard.jsonl"

    shard.write_text(
        '{"example_id":"e1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="completeness failure",
    ):
        reduce_shard_jsonl(
            [shard],
            tmp_path / "out.jsonl",
            {"e1", "e2"},
        )


def test_extra_results_are_rejected(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard.jsonl"

    shard.write_text(
        ('{"example_id":"e1"}\n{"example_id":"unexpected"}\n'),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="completeness failure",
    ):
        reduce_shard_jsonl(
            [shard],
            tmp_path / "out.jsonl",
            {"e1"},
        )


def test_missing_example_id_is_rejected(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard.jsonl"

    shard.write_text(
        '{"ok":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing example_id",
    ):
        reduce_shard_jsonl(
            [shard],
            tmp_path / "out.jsonl",
            {"e1"},
        )
