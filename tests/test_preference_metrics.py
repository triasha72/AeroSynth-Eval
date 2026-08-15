from aerosynth_eval.pairwise_judge import JudgeOrientation, PairwiseJudgmentRecord
from aerosynth_eval.preference_benchmark import PreferenceLabel
from aerosynth_eval.preference_metrics import summarize_preference_metrics


def _record(
    example: str, orientation: JudgeOrientation, human: PreferenceLabel, predicted: PreferenceLabel
) -> PairwiseJudgmentRecord:
    return PairwiseJudgmentRecord(
        example_id=example,
        orientation=orientation,
        model_id="judge",
        human_preference=human,
        predicted_preference=predicted,
        confidence=0.8,
        rationale="test",
        raw_output="{}",
    )


def test_metrics_and_swap_consistency() -> None:
    records = [
        _record(
            "e1",
            JudgeOrientation.STANDARD,
            PreferenceLabel.A_PREFERRED,
            PreferenceLabel.A_PREFERRED,
        ),
        _record(
            "e1", JudgeOrientation.SWAPPED, PreferenceLabel.B_PREFERRED, PreferenceLabel.B_PREFERRED
        ),
        _record(
            "e2", JudgeOrientation.STANDARD, PreferenceLabel.TIE_GOOD, PreferenceLabel.TIE_GOOD
        ),
    ]
    summary = summarize_preference_metrics(records)
    assert summary.accuracy == 1.0
    assert summary.swap_consistency == 1.0
