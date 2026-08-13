"""Versioned evaluation rubric for synthetic aerospace inspection imagery."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.contracts import EvaluationDimension

RUBRIC_VERSION = "v0.1"


class RubricDimension(BaseModel):
    """Describes one fixed scoring dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: EvaluationDimension
    question: str
    score_0: str
    score_4: str


class InspectionRubric(BaseModel):
    """The fixed, reviewable rubric used in the foundation release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    score_scale: dict[int, str]
    dimensions: tuple[RubricDimension, ...] = Field(min_length=4, max_length=4)


def inspection_rubric() -> InspectionRubric:
    """Return the fixed v0.1 rubric before any prompt or model iteration."""

    return InspectionRubric(
        version=RUBRIC_VERSION,
        score_scale={
            0: "Absent, unusable, or clearly contradictory.",
            1: "Weak evidence with major deficiencies.",
            2: "Partially satisfied with material limitations.",
            3: "Mostly satisfied with minor limitations.",
            4: "Fully satisfied and clearly usable.",
        },
        dimensions=(
            RubricDimension(
                dimension=EvaluationDimension.CONTEXT_FIDELITY,
                question=(
                    "Does the image depict the requested aircraft region and inspection context?"
                ),
                score_0="The requested component or context is absent or incorrect.",
                score_4="The requested component and inspection context are clearly depicted.",
            ),
            RubricDimension(
                dimension=EvaluationDimension.CONDITION_FIDELITY,
                question="Does the visible condition match the requested condition?",
                score_0="The requested condition is absent, contradictory, or indiscernible.",
                score_4=(
                    "The requested condition is clearly visible and consistent with "
                    "the specification."
                ),
            ),
            RubricDimension(
                dimension=EvaluationDimension.IMAGE_QUALITY,
                question=(
                    "Is the image sufficiently focused, well-lit, and free of "
                    "obstructive artifacts?"
                ),
                score_0="Blur, glare, occlusion, or artifacts make the image unusable.",
                score_4="Focus, lighting, framing, and visual clarity are all suitable for review.",
            ),
            RubricDimension(
                dimension=EvaluationDimension.INSPECTION_UTILITY,
                question=(
                    "Could the image plausibly support a non-operational inspection-data workflow?"
                ),
                score_0="The image is not suitable for dataset review or research use.",
                score_4=(
                    "The image is suitable for research dataset review, with the "
                    "stated limitations."
                ),
            ),
        ),
    )
