import json
from pathlib import Path

import pytest

from aerosynth_eval.asset_registry import (
    load_asset_registry,
    summarize_asset_registry,
    validate_asset_registry,
)
from aerosynth_eval.contracts import AssetLifecycleStatus, DatasetSplit
from aerosynth_eval.scenario_matrix import load_scenario_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"


def test_load_asset_registry_reads_all_planned_assets() -> None:
    records = load_asset_registry(REGISTRY_PATH)

    assert len(records) == 48
    assert records[0].asset_id == "asset-fuselage-clean-close-diffuse"
    assert records[-1].lifecycle_status is AssetLifecycleStatus.PLANNED


def test_validate_asset_registry_matches_frozen_scenario_matrix() -> None:
    summary = validate_asset_registry(
        load_asset_registry(REGISTRY_PATH),
        load_scenario_matrix(MATRIX_PATH),
    )

    assert summary.splits[DatasetSplit.DEVELOPMENT] == 36
    assert summary.splits[DatasetSplit.TEST] == 12
    assert summary.lifecycle_statuses[AssetLifecycleStatus.PLANNED] == 48
    assert summary.records_with_generation_evidence == 0


def test_load_asset_registry_rejects_duplicate_scenario_ids(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8").splitlines()[0])
    duplicate_payload = {
        **payload,
        "asset_id": "asset-duplicate-scenario",
        "image_reference": "assets/v0_1/duplicate-scenario.png",
    }
    duplicate_registry = tmp_path / "duplicate.jsonl"
    duplicate_registry.write_text(
        f"{json.dumps(payload)}\n{json.dumps(duplicate_payload)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate scenario_id"):
        load_asset_registry(duplicate_registry)


def test_load_asset_registry_rejects_generated_record_without_evidence(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8").splitlines()[0])
    payload["lifecycle_status"] = "generated"
    invalid_registry = tmp_path / "missing-evidence.jsonl"
    invalid_registry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generated asset records require generation evidence"):
        load_asset_registry(invalid_registry)


def test_summarize_asset_registry_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="empty asset registry"):
        summarize_asset_registry(())
