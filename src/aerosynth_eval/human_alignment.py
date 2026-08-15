"""Compare evaluator outputs against an adjudicated human development reference."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HumanReferenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    decision: str
    scores: dict[str, int]


class EvaluatorCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    decision: str
    scores: dict[str, int]
    confidence: float = Field(ge=0.0, le=1.0)


class HumanAlignmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_count: int = Field(ge=1)
    decision_agreement: float = Field(ge=0.0, le=1.0)
    mean_dimension_mae: dict[str, float]
    high_confidence_error_rate: float = Field(ge=0.0, le=1.0)


def summarize_human_alignment(
    human: list[HumanReferenceCase], evaluator: list[EvaluatorCase]
) -> HumanAlignmentSummary:
    human_by_id = {record.scenario_id: record for record in human}
    evaluator_by_id = {record.scenario_id: record for record in evaluator}
    if set(human_by_id) != set(evaluator_by_id) or not human_by_id:
        raise ValueError("Human and evaluator cases must cover the same non-empty scenario set.")
    scenario_ids = sorted(human_by_id)
    agreement = sum(
        human_by_id[s].decision == evaluator_by_id[s].decision for s in scenario_ids
    ) / len(scenario_ids)
    dimensions = sorted(set.intersection(*(set(human_by_id[s].scores) for s in scenario_ids)))
    mae = {
        dimension: sum(
            abs(human_by_id[s].scores[dimension] - evaluator_by_id[s].scores[dimension])
            for s in scenario_ids
        )
        / len(scenario_ids)
        for dimension in dimensions
    }
    high_confidence = [s for s in scenario_ids if evaluator_by_id[s].confidence >= 0.8]
    errors = [s for s in high_confidence if human_by_id[s].decision != evaluator_by_id[s].decision]
    return HumanAlignmentSummary(
        case_count=len(scenario_ids),
        decision_agreement=agreement,
        mean_dimension_mae=mae,
        high_confidence_error_rate=len(errors) / len(high_confidence) if high_confidence else 0.0,
    )
