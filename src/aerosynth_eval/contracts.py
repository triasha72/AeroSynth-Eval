"""Typed contracts for synthetic-image evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AircraftRegion(StrEnum):
    """Aircraft exterior regions supported by the initial rubric."""

    FUSELAGE_PANEL = "fuselage_panel"
    WING_PANEL = "wing_panel"
    EMPENNAGE_PANEL = "empennage_panel"


class SurfaceCondition(StrEnum):
    """Surface conditions supported by the initial rubric."""

    NO_VISIBLE_DEFECT = "no_visible_defect"
    CORROSION = "corrosion"
    SURFACE_CRACK = "surface_crack"
    COATING_DAMAGE = "coating_damage"


class EvaluationDimension(StrEnum):
    """Required dimensions for each evaluation."""

    CONTEXT_FIDELITY = "context_fidelity"
    CONDITION_FIDELITY = "condition_fidelity"
    IMAGE_QUALITY = "image_quality"
    INSPECTION_UTILITY = "inspection_utility"


class EvaluationDecision(StrEnum):
    """Top-level disposition assigned by an evaluator."""

    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class InspectionSpecification(BaseModel):
    """Expected properties of one synthetic inspection image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: AircraftRegion
    condition: SurfaceCondition
    viewpoint: str = Field(min_length=3, max_length=120)
    lighting: str = Field(min_length=3, max_length=120)
    required_attributes: tuple[str, ...] = Field(min_length=1, max_length=10)

    @field_validator("required_attributes")
    @classmethod
    def normalize_required_attributes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Remove blank attributes and reject an empty final list."""

        normalized = tuple(value.strip() for value in values if value.strip())
        if not normalized:
            raise ValueError("required_attributes must contain at least one non-empty value.")
        return normalized


class SyntheticImageExample(BaseModel):
    """A generated image and the specification it is expected to satisfy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    prompt: str = Field(min_length=8, max_length=2_000)
    image_reference: str = Field(min_length=1, max_length=1_000)
    specification: InspectionSpecification


class DimensionScore(BaseModel):
    """One rubric score emitted by an autograder or human rater."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: EvaluationDimension
    score: int = Field(ge=0, le=4)
    rationale: str = Field(min_length=1, max_length=500)


class AutograderResult(BaseModel):
    """Structured result contract for future baseline and VLM evaluators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str
    rubric_version: str
    decision: EvaluationDecision
    confidence: float = Field(ge=0.0, le=1.0)
    scores: tuple[DimensionScore, ...] = Field(min_length=4, max_length=4)
    summary: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def require_each_rubric_dimension_once(self) -> Self:
        """Ensure every result has exactly one score per rubric dimension."""

        dimensions = {score.dimension for score in self.scores}
        expected_dimensions = set(EvaluationDimension)
        if dimensions != expected_dimensions:
            raise ValueError("scores must contain each evaluation dimension exactly once.")
        return self
