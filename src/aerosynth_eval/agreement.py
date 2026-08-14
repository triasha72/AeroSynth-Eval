"""Agreement summaries for explicitly synthetic demo fixtures."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.annotations import RaterAnnotationRecord
from aerosynth_eval.contracts import EvaluationDecision, EvaluationDimension

SYNTHETIC_DEMO_RATER_PREFIX: Final = "synthetic_demo_"
SYNTHETIC_DEMO_RATIONALE_PREFIX: Final = "Synthetic demo only:"
DIMENSIONS: Final[tuple[EvaluationDimension, ...]] = tuple(EvaluationDimension)


class AgreementRate(BaseModel):
    """A count, denominator, and rate for one agreement measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int = Field(ge=0)
    total: int = Field(ge=1)
    rate: float = Field(ge=0.0, le=1.0)


class DimensionAgreementSummary(BaseModel):
    """Agreement measurements for one ordinal rubric dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exact_agreement: AgreementRate
    within_one_agreement: AgreementRate
    mean_absolute_difference: float = Field(ge=0.0, le=4.0)


class DisagreementCandidate(BaseModel):
    """A synthetic fixture pair that should appear in disagreement reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    decision_by_rater: dict[str, EvaluationDecision]
    score_differences: dict[str, int]
    maximum_score_difference: int = Field(ge=0, le=4)


class SyntheticAgreementSummary(BaseModel):
    """A strictly synthetic-fixture agreement summary, not human-study evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_classification: str = Field(pattern=r"^synthetic_demo_fixture$")
    claim_scope: str = Field(pattern=r"^software_test_fixture_only$")
    human_agreement_claim_supported: bool = False
    record_count: int = Field(ge=1)
    rater_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    decision_exact_agreement: AgreementRate
    dimensions: dict[str, DimensionAgreementSummary]
    disagreement_candidates: tuple[DisagreementCandidate, ...]


def _rate(count: int, total: int) -> AgreementRate:
    """Build a validated rate with one shared denominator."""

    return AgreementRate(count=count, total=total, rate=count / total)


def _dimension_scores(record: RaterAnnotationRecord) -> dict[EvaluationDimension, int]:
    """Return all fixed rubric scores from a rater record."""

    return {
        EvaluationDimension.CONTEXT_FIDELITY: record.context_fidelity,
        EvaluationDimension.CONDITION_FIDELITY: record.condition_fidelity,
        EvaluationDimension.IMAGE_QUALITY: record.image_quality,
        EvaluationDimension.INSPECTION_UTILITY: record.inspection_utility,
    }


def _require_synthetic_demo_submission(
    records: tuple[RaterAnnotationRecord, ...],
    source_label: str,
) -> str:
    """Reject anything that is not one complete, clearly marked synthetic fixture."""

    if not records:
        raise ValueError(f"{source_label} synthetic demo submission is empty.")

    rater_ids = {record.rater_id for record in records}
    if len(rater_ids) != 1:
        raise ValueError(
            f"{source_label} synthetic demo submission must contain exactly one rater_id."
        )
    rater_id = next(iter(rater_ids))
    if not rater_id.startswith(SYNTHETIC_DEMO_RATER_PREFIX):
        raise ValueError(
            f"{source_label} rater_id must begin with '{SYNTHETIC_DEMO_RATER_PREFIX}'."
        )

    for record in records:
        rationales = (
            record.decision_rationale,
            record.context_fidelity_rationale,
            record.condition_fidelity_rationale,
            record.image_quality_rationale,
            record.inspection_utility_rationale,
        )
        if any(
            not rationale.startswith(SYNTHETIC_DEMO_RATIONALE_PREFIX) for rationale in rationales
        ):
            raise ValueError(
                f"{source_label} contains a rationale that is not marked as a synthetic demo."
            )
    return rater_id


def _records_by_scenario(
    records: tuple[RaterAnnotationRecord, ...],
    source_label: str,
) -> dict[str, RaterAnnotationRecord]:
    """Index a submission by scenario while rejecting duplicate scenarios."""

    indexed = {record.scenario_id: record for record in records}
    if len(indexed) != len(records):
        raise ValueError(
            f"{source_label} synthetic demo submission contains duplicate scenario_id values."
        )
    return indexed


def summarize_synthetic_demo_agreement(
    first_records: tuple[RaterAnnotationRecord, ...],
    second_records: tuple[RaterAnnotationRecord, ...],
) -> SyntheticAgreementSummary:
    """Summarize a pair of explicitly synthetic demo submissions.

    This function intentionally accepts only fixture records marked with the
    synthetic-demo rater-ID and rationale conventions. It is software-test
    plumbing and must not be used to produce human-agreement claims.
    """

    first_rater_id = _require_synthetic_demo_submission(first_records, "First")
    second_rater_id = _require_synthetic_demo_submission(second_records, "Second")
    if first_rater_id == second_rater_id:
        raise ValueError("Synthetic demo submissions must use distinct rater_id values.")

    first_by_scenario = _records_by_scenario(first_records, "First")
    second_by_scenario = _records_by_scenario(second_records, "Second")
    if set(first_by_scenario) != set(second_by_scenario):
        raise ValueError("Synthetic demo submissions must cover the same scenario_id values.")

    scenario_ids = tuple(sorted(first_by_scenario))
    total = len(scenario_ids)
    decision_exact_count = 0
    dimension_differences: dict[EvaluationDimension, list[int]] = {
        dimension: [] for dimension in DIMENSIONS
    }
    disagreement_candidates: list[DisagreementCandidate] = []

    for scenario_id in scenario_ids:
        first_record = first_by_scenario[scenario_id]
        second_record = second_by_scenario[scenario_id]
        if first_record.decision == second_record.decision:
            decision_exact_count += 1

        first_scores = _dimension_scores(first_record)
        second_scores = _dimension_scores(second_record)
        score_differences = {
            dimension.value: abs(first_scores[dimension] - second_scores[dimension])
            for dimension in DIMENSIONS
        }
        for dimension in DIMENSIONS:
            dimension_differences[dimension].append(score_differences[dimension.value])

        maximum_score_difference = max(score_differences.values())
        if first_record.decision != second_record.decision or maximum_score_difference >= 2:
            disagreement_candidates.append(
                DisagreementCandidate(
                    scenario_id=scenario_id,
                    decision_by_rater={
                        first_rater_id: first_record.decision,
                        second_rater_id: second_record.decision,
                    },
                    score_differences=score_differences,
                    maximum_score_difference=maximum_score_difference,
                )
            )

    dimensions: dict[str, DimensionAgreementSummary] = {}
    for dimension in DIMENSIONS:
        differences = dimension_differences[dimension]
        exact_count = sum(difference == 0 for difference in differences)
        within_one_count = sum(difference <= 1 for difference in differences)
        dimensions[dimension.value] = DimensionAgreementSummary(
            exact_agreement=_rate(exact_count, total),
            within_one_agreement=_rate(within_one_count, total),
            mean_absolute_difference=sum(differences) / total,
        )

    return SyntheticAgreementSummary(
        data_classification="synthetic_demo_fixture",
        claim_scope="software_test_fixture_only",
        record_count=total,
        rater_ids=tuple(sorted((first_rater_id, second_rater_id))),
        decision_exact_agreement=_rate(decision_exact_count, total),
        dimensions=dimensions,
        disagreement_candidates=tuple(disagreement_candidates),
    )
