import pytest

from aerosynth_eval.experiment_hillclimbing import (
    ExperimentResult,
    rank_experiments,
)
from aerosynth_eval.preference_benchmark import (
    BenchmarkPartition,
)


def test_experiment_ranking() -> None:
    weak = ExperimentResult(
        experiment_id="a",
        judge_id="j",
        prompt_version="p",
        partition=BenchmarkPartition.VALIDATION,
        human_agreement=0.5,
        macro_f1=0.5,
        swap_consistency=0.5,
        ece=0.3,
        failure_rate=0.1,
    )
    strong = ExperimentResult(
        experiment_id="b",
        judge_id="j",
        prompt_version="p",
        partition=BenchmarkPartition.VALIDATION,
        human_agreement=0.8,
        macro_f1=0.8,
        swap_consistency=0.9,
        ece=0.1,
        failure_rate=0.0,
    )
    assert rank_experiments([weak, strong])[0].experiment_id == "b"


def test_experiment_ranking_rejects_heldout() -> None:
    heldout = ExperimentResult(
        experiment_id="heldout-run",
        judge_id="j",
        prompt_version="p",
        partition=BenchmarkPartition.HELDOUT,
        human_agreement=0.9,
        macro_f1=0.9,
        swap_consistency=0.9,
        ece=0.05,
        failure_rate=0.0,
    )

    with pytest.raises(
        ValueError,
        match="restricted to validation",
    ):
        rank_experiments([heldout])


def test_experiment_ranking_rejects_train() -> None:
    train = ExperimentResult(
        experiment_id="train-run",
        judge_id="j",
        prompt_version="p",
        partition=BenchmarkPartition.TRAIN,
        human_agreement=0.9,
        macro_f1=0.9,
        swap_consistency=0.9,
        ece=0.05,
        failure_rate=0.0,
    )

    with pytest.raises(
        ValueError,
        match="restricted to validation",
    ):
        rank_experiments([train])
