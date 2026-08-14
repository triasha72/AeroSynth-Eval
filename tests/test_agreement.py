from pathlib import Path

import pytest

from aerosynth_eval.agreement import summarize_synthetic_demo_agreement
from aerosynth_eval.annotations import load_rater_annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_demo"
FIXTURE_A = FIXTURE_ROOT / "synthetic_demo_rater_a.csv"
FIXTURE_B = FIXTURE_ROOT / "synthetic_demo_rater_b.csv"


def test_synthetic_fixture_summary_is_explicitly_non_human() -> None:
    summary = summarize_synthetic_demo_agreement(
        load_rater_annotations(FIXTURE_A),
        load_rater_annotations(FIXTURE_B),
    )

    assert summary.data_classification == "synthetic_demo_fixture"
    assert summary.claim_scope == "software_test_fixture_only"
    assert summary.human_agreement_claim_supported is False
    assert summary.record_count == 12
    assert summary.rater_ids == ("synthetic_demo_a", "synthetic_demo_b")
    assert summary.decision_exact_agreement.count == 10
    assert summary.decision_exact_agreement.rate == pytest.approx(10 / 12)
    assert summary.dimensions["context_fidelity"].exact_agreement.count == 11
    assert summary.dimensions["condition_fidelity"].exact_agreement.count == 10
    assert summary.dimensions["image_quality"].exact_agreement.count == 11
    assert summary.dimensions["inspection_utility"].exact_agreement.count == 7
    assert summary.dimensions["inspection_utility"].mean_absolute_difference == pytest.approx(
        5 / 12
    )
    assert {candidate.scenario_id for candidate in summary.disagreement_candidates} == {
        "empennage-coating-oblique-blur",
        "fuselage-coating-close-glare",
    }


def test_synthetic_fixture_summary_rejects_non_synthetic_rater_ids() -> None:
    first_records = load_rater_annotations(FIXTURE_A)
    non_synthetic_records = tuple(
        record.model_copy(update={"rater_id": "rater_a"}) for record in first_records
    )

    with pytest.raises(ValueError, match="synthetic_demo_"):
        summarize_synthetic_demo_agreement(
            non_synthetic_records,
            load_rater_annotations(FIXTURE_B),
        )


def test_synthetic_fixture_summary_rejects_duplicate_rater_ids() -> None:
    first_records = load_rater_annotations(FIXTURE_A)
    second_records = tuple(
        record.model_copy(update={"rater_id": "synthetic_demo_a"})
        for record in load_rater_annotations(FIXTURE_B)
    )

    with pytest.raises(ValueError, match="distinct"):
        summarize_synthetic_demo_agreement(first_records, second_records)
