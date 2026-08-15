from aerosynth_eval.human_alignment import (
    EvaluatorCase,
    HumanReferenceCase,
    summarize_human_alignment,
)


def test_human_alignment() -> None:
    human = [HumanReferenceCase(scenario_id="s", decision="accept", scores={"image_quality": 4})]
    evaluator = [
        EvaluatorCase(
            scenario_id="s", decision="accept", scores={"image_quality": 3}, confidence=0.9
        )
    ]
    summary = summarize_human_alignment(human, evaluator)
    assert summary.decision_agreement == 1.0
    assert summary.mean_dimension_mae["image_quality"] == 1.0
