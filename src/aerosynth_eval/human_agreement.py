"""Real human-human agreement metrics for completed private rater submissions."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DIMENSIONS = ("context_fidelity", "condition_fidelity", "image_quality", "inspection_utility")


class OrdinalAgreement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exact_agreement: float = Field(ge=0.0, le=1.0)
    within_one_agreement: float = Field(ge=0.0, le=1.0)
    mean_absolute_difference: float = Field(ge=0.0)
    quadratic_weighted_kappa: float = Field(ge=-1.0, le=1.0)


class HumanAgreementSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_count: int = Field(ge=1)
    decision_exact_agreement: float = Field(ge=0.0, le=1.0)
    decision_kappa: float = Field(ge=-1.0, le=1.0)
    dimensions: dict[str, OrdinalAgreement]
    disagreement_scenarios: list[str]


def _load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["scenario_id"]: row for row in rows}


def _kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    expected = sum((a.count(label) / len(a)) * (b.count(label) / len(b)) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _quadratic_weighted_kappa(
    a: list[int], b: list[int], minimum: int = 0, maximum: int = 4
) -> float:
    n = len(a)
    categories = list(range(minimum, maximum + 1))
    observed = [[0.0 for _ in categories] for _ in categories]
    for left, right in zip(a, b, strict=True):
        observed[left - minimum][right - minimum] += 1.0 / n
    hist_a = [a.count(category) / n for category in categories]
    hist_b = [b.count(category) / n for category in categories]
    weighted_observed = 0.0
    weighted_expected = 0.0
    denominator = float((maximum - minimum) ** 2) or 1.0
    for i, _ in enumerate(categories):
        for j, _ in enumerate(categories):
            weight = ((i - j) ** 2) / denominator
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * hist_a[i] * hist_b[j]
    return 1.0 - weighted_observed / weighted_expected if weighted_expected else 1.0


def summarize_human_agreement(path_a: Path, path_b: Path) -> HumanAgreementSummary:
    a = _load(path_a)
    b = _load(path_b)
    if set(a) != set(b) or not a:
        raise ValueError("Rater submissions must cover the same non-empty scenario set.")
    scenarios = sorted(a)
    decisions_a = [a[s]["decision"] for s in scenarios]
    decisions_b = [b[s]["decision"] for s in scenarios]
    dimensions: dict[str, OrdinalAgreement] = {}
    for dimension in DIMENSIONS:
        scores_a = [int(a[s][dimension]) for s in scenarios]
        scores_b = [int(b[s][dimension]) for s in scenarios]
        differences = [abs(x - y) for x, y in zip(scores_a, scores_b, strict=True)]
        dimensions[dimension] = OrdinalAgreement(
            exact_agreement=sum(x == y for x, y in zip(scores_a, scores_b, strict=True))
            / len(scenarios),
            within_one_agreement=sum(diff <= 1 for diff in differences) / len(scenarios),
            mean_absolute_difference=sum(differences) / len(differences),
            quadratic_weighted_kappa=_quadratic_weighted_kappa(scores_a, scores_b),
        )
    disagreements = [
        s
        for s in scenarios
        if any(a[s].get(key) != b[s].get(key) for key in ("decision", *DIMENSIONS))
    ]
    return HumanAgreementSummary(
        scenario_count=len(scenarios),
        decision_exact_agreement=sum(x == y for x, y in zip(decisions_a, decisions_b, strict=True))
        / len(scenarios),
        decision_kappa=_kappa(decisions_a, decisions_b),
        dimensions=dimensions,
        disagreement_scenarios=disagreements,
    )
