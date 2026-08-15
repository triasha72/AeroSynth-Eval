from pathlib import Path

from aerosynth_eval.scalable_evaluation import JsonlCheckpointStore, cache_key, shard_index


def test_sharding_and_cache_are_deterministic(tmp_path: Path) -> None:
    assert shard_index("abc", 8) == shard_index("abc", 8)
    assert cache_key(
        example_id="e", model_id="m", model_revision="r", prompt_version="p", config_payload="x"
    ) == cache_key(
        example_id="e", model_id="m", model_revision="r", prompt_version="p", config_payload="x"
    )
    store = JsonlCheckpointStore(tmp_path / "checkpoint.jsonl")
    assert store.completed_ids() == set()
    store.append_success("e1", {"ok": True})
    assert store.completed_ids() == {"e1"}
