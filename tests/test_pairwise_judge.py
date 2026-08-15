import json
from pathlib import Path

from aerosynth_eval.pairwise_judge import JudgeOrientation, judge_example, swap_label
from aerosynth_eval.preference_benchmark import (
    BenchmarkPartition,
    PreferenceExample,
    PreferenceLabel,
    PreferenceTask,
)


def _example() -> PreferenceExample:
    return PreferenceExample(
        example_id="genai-0123456789abcdef",
        task=PreferenceTask.IMAGE_GENERATION,
        partition=BenchmarkPartition.HELDOUT,
        prompt="a red airplane",
        left_model="m1",
        right_model="m2",
        human_preference=PreferenceLabel.A_PREFERRED,
        left_image="left.png",
        right_image="right.png",
    )


def test_injected_pairwise_judge() -> None:
    def infer(prompt: str, left: Path, right: Path) -> str:
        assert prompt
        assert left.name == "left.png"
        assert right.name == "right.png"
        return json.dumps({"preference": "a_preferred", "confidence": 0.8, "rationale": "better"})

    record = judge_example(_example(), model_id="test", inference=infer)
    assert record.predicted_preference is PreferenceLabel.A_PREFERRED


def test_swap_label() -> None:
    assert swap_label(PreferenceLabel.A_PREFERRED) is PreferenceLabel.B_PREFERRED
    assert swap_label(PreferenceLabel.TIE_GOOD) is PreferenceLabel.TIE_GOOD


def test_swapped_orientation_uses_presented_position_labels() -> None:
    example = _example()

    def infer(prompt: str, left: Path, right: Path) -> str:
        assert left.name == "right.png"
        assert right.name == "left.png"
        assert "Candidate A is always the first presented image" in prompt
        assert "candidate B is always the second presented image" in prompt
        return json.dumps(
            {
                "preference": "b_preferred",
                "confidence": 0.9,
                "rationale": "The second presented image better satisfies the request.",
            }
        )

    record = judge_example(
        example,
        model_id="test",
        inference=infer,
        orientation=JudgeOrientation.SWAPPED,
    )

    assert record.human_preference is PreferenceLabel.B_PREFERRED
    assert record.predicted_preference is PreferenceLabel.B_PREFERRED
