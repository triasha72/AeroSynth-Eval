from pathlib import Path

import pytest

from aerosynth_eval.preference_benchmark import (
    BenchmarkPartition,
    PreferenceLabel,
    PreferenceTask,
    map_genai_vote,
    normalize_record,
    partition_for_group,
    summarize_manifest,
)


def test_vote_mapping_and_partition_are_stable() -> None:
    assert map_genai_vote("leftvote") is PreferenceLabel.A_PREFERRED
    assert map_genai_vote("rightvote") is PreferenceLabel.B_PREFERRED
    assert map_genai_vote("tievote") is PreferenceLabel.TIE_GOOD
    assert map_genai_vote("bothbad_vote") is PreferenceLabel.TIE_BAD
    assert partition_for_group("same prompt") is partition_for_group("same prompt")


def test_normalize_image_generation_record() -> None:
    record = {
        "prompt": "a test prompt",
        "left_model": "left",
        "right_model": "right",
        "vote_type": "leftvote",
    }
    example = normalize_record(
        record,
        task=PreferenceTask.IMAGE_GENERATION,
        source_index=1,
        left_image_path=Path("left.png"),
        right_image_path=Path("right.png"),
    )
    assert example.human_preference is PreferenceLabel.A_PREFERRED
    assert example.partition in set(BenchmarkPartition)
    summary = summarize_manifest([example])
    assert summary.record_count == 1


def test_unknown_vote_rejected() -> None:
    with pytest.raises(ValueError):
        map_genai_vote("mystery")
