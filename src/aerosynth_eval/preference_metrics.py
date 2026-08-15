"""Judge-human agreement metrics, bootstrap uncertainty, and order-bias analysis."""

from __future__ import annotations

import random
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.pairwise_judge import JudgeOrientation, PairwiseJudgmentRecord, swap_label
from aerosynth_eval.preference_benchmark import PreferenceLabel


class MetricInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    estimate: float
    lower_95: float
    upper_95: float


class PreferenceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_precision: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    tie_accuracy: float | None
    non_tie_accuracy: float | None
    swap_consistency: float | None
    position_bias_rate: float | None
    accuracy_bootstrap: MetricInterval
    confusion: dict[str, dict[str, int]]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _accuracy(records: list[PairwiseJudgmentRecord]) -> float:
    return _safe_div(
        sum(r.predicted_preference is r.human_preference for r in records), len(records)
    )


def bootstrap_accuracy(
    records: list[PairwiseJudgmentRecord], *, samples: int = 2_000, seed: int = 0
) -> MetricInterval:
    if not records:
        raise ValueError("At least one judgment is required.")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [records[rng.randrange(len(records))] for _ in records]
        estimates.append(_accuracy(sample))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return MetricInterval(estimate=_accuracy(records), lower_95=lower, upper_95=upper)


def summarize_preference_metrics(records: list[PairwiseJudgmentRecord]) -> PreferenceMetrics:
    if not records:
        raise ValueError("At least one pairwise judgment is required.")
    labels = list(PreferenceLabel)
    confusion: dict[str, dict[str, int]] = {
        label.value: {pred.value: 0 for pred in labels} for label in labels
    }
    for record in records:
        confusion[record.human_preference.value][record.predicted_preference.value] += 1

    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    for label in labels:
        true_positive = confusion[label.value][label.value]
        actual = sum(confusion[label.value].values())
        predicted = sum(confusion[actual_label.value][label.value] for actual_label in labels)
        recall = _safe_div(true_positive, actual)
        precision = _safe_div(true_positive, predicted)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    ties = [
        r
        for r in records
        if r.human_preference in {PreferenceLabel.TIE_GOOD, PreferenceLabel.TIE_BAD}
    ]
    non_ties = [
        r
        for r in records
        if r.human_preference in {PreferenceLabel.A_PREFERRED, PreferenceLabel.B_PREFERRED}
    ]

    by_example: dict[str, dict[JudgeOrientation, PairwiseJudgmentRecord]] = defaultdict(dict)
    for record in records:
        by_example[record.example_id][record.orientation] = record
    swap_pairs = [pair for pair in by_example.values() if len(pair) == 2]
    consistent = 0
    for pair in swap_pairs:
        standard = pair[JudgeOrientation.STANDARD]
        swapped = pair[JudgeOrientation.SWAPPED]
        if swap_label(standard.predicted_preference) is swapped.predicted_preference:
            consistent += 1
    swap_consistency = _safe_div(consistent, len(swap_pairs)) if swap_pairs else None

    return PreferenceMetrics(
        record_count=len(records),
        accuracy=_accuracy(records),
        balanced_accuracy=sum(recalls) / len(recalls),
        macro_precision=sum(precisions) / len(precisions),
        macro_recall=sum(recalls) / len(recalls),
        macro_f1=sum(f1s) / len(f1s),
        tie_accuracy=_accuracy(ties) if ties else None,
        non_tie_accuracy=_accuracy(non_ties) if non_ties else None,
        swap_consistency=swap_consistency,
        position_bias_rate=(1.0 - swap_consistency) if swap_consistency is not None else None,
        accuracy_bootstrap=bootstrap_accuracy(records),
        confusion=confusion,
    )
