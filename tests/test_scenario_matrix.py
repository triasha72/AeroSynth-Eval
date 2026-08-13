from pathlib import Path

import pytest

from aerosynth_eval.contracts import AircraftRegion, DatasetSplit, SurfaceCondition
from aerosynth_eval.scenario_matrix import (
    load_scenario_matrix,
    summarize_scenario_matrix,
    validate_scenario_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"


def test_load_scenario_matrix_reads_balanced_fixture() -> None:
    records = load_scenario_matrix(MATRIX_PATH)

    assert len(records) == 48
    assert records[0].scenario_id == "fuselage-clean-close-diffuse"
    assert records[-1].split is DatasetSplit.DEVELOPMENT


def test_validate_scenario_matrix_enforces_v0_1_balance() -> None:
    summary = validate_scenario_matrix(load_scenario_matrix(MATRIX_PATH))

    assert summary.splits[DatasetSplit.DEVELOPMENT] == 36
    assert summary.splits[DatasetSplit.TEST] == 12
    assert summary.components[AircraftRegion.FUSELAGE_PANEL] == 16
    assert summary.conditions[SurfaceCondition.CORROSION] == 12
    assert summary.capture_profiles["close_glare"] == 12


def test_load_scenario_matrix_rejects_duplicate_ids(tmp_path: Path) -> None:
    lines = MATRIX_PATH.read_text(encoding="utf-8").splitlines()
    duplicate_matrix = tmp_path / "duplicate.csv"
    duplicate_matrix.write_text("\n".join([lines[0], lines[1], lines[1]]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate scenario_id"):
        load_scenario_matrix(duplicate_matrix)


def test_summarize_scenario_matrix_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="empty scenario matrix"):
        summarize_scenario_matrix(())
