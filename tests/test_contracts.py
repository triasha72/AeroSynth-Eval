import pytest
from pydantic import ValidationError

from aerosynth_eval.contracts import (
    AircraftRegion,
    DimensionScore,
    EvaluationDecision,
    EvaluationDimension,
    HumanAnnotation,
    InspectionSpecification,
    SurfaceCondition,
    SyntheticImageExample,
)


def test_synthetic_image_example_accepts_valid_specification() -> None:
    example = SyntheticImageExample(
        example_id="fuselage-corrosion-001",
        prompt="Close inspection view of a fuselage panel with localized corrosion.",
        image_reference="data/raw/fuselage-corrosion-001.png",
        specification=InspectionSpecification(
            component=AircraftRegion.FUSELAGE_PANEL,
            condition=SurfaceCondition.CORROSION,
            viewpoint="close inspection view",
            lighting="diffuse daylight",
            required_attributes=("metal surface", "localized corrosion", "no people"),
        ),
    )

    assert example.specification.condition is SurfaceCondition.CORROSION
    assert example.specification.required_attributes == (
        "metal surface",
        "localized corrosion",
        "no people",
    )


def test_synthetic_image_example_rejects_invalid_identifier() -> None:
    with pytest.raises(ValidationError):
        SyntheticImageExample(
            example_id="X",
            prompt="Close inspection view of an aircraft fuselage panel.",
            image_reference="data/raw/example.png",
            specification=InspectionSpecification(
                component=AircraftRegion.FUSELAGE_PANEL,
                condition=SurfaceCondition.NO_VISIBLE_DEFECT,
                viewpoint="close inspection view",
                lighting="diffuse daylight",
                required_attributes=("metal surface",),
            ),
        )


def test_human_annotation_accepts_one_score_for_each_dimension() -> None:
    annotation = HumanAnnotation(
        annotation_id="annotation-001",
        example_id="fuselage-corrosion-001",
        rater_id="rater_01",
        rubric_version="v0.1",
        decision=EvaluationDecision.UNCERTAIN,
        scores=tuple(
            DimensionScore(dimension=dimension, score=2, rationale="Evidence is partial.")
            for dimension in EvaluationDimension
        ),
        notes="Escalate because condition visibility is ambiguous.",
    )

    assert annotation.decision is EvaluationDecision.UNCERTAIN
