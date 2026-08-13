import hashlib
import json
from pathlib import Path

import pytest

from aerosynth_eval.asset_registry import (
    load_asset_registry,
    validate_materialized_assets,
)
from aerosynth_eval.procedural_corpus import materialize_procedural_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "design" / "v0_1_scenario_matrix.csv"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "v0_1_asset_registry.jsonl"


def _write_planned_registry(target_directory: Path) -> Path:
    target_directory.mkdir(parents=True, exist_ok=True)
    registry_path = target_directory / "registry.jsonl"
    planned_records: list[dict[str, object]] = []
    for record in load_asset_registry(REGISTRY_PATH):
        payload = record.model_dump(mode="json")
        payload["lifecycle_status"] = "planned"
        for field_name in (
            "generator_name",
            "generator_model",
            "generator_version",
            "generation_prompt",
            "seed",
            "generated_at",
            "image_sha256",
        ):
            payload[field_name] = None
        payload["source_note"] = (
            "Reserved v0.1 synthetic asset path; no image has been generated or distributed."
        )
        planned_records.append(payload)
    registry_path.write_text(
        "\n".join(json.dumps(record) for record in planned_records) + "\n",
        encoding="utf-8",
    )
    return registry_path


def test_materialize_procedural_corpus_generates_and_validates_all_assets(tmp_path: Path) -> None:
    registry_path = _write_planned_registry(tmp_path)
    asset_root = tmp_path / "data"

    summary = materialize_procedural_corpus(registry_path, MATRIX_PATH, asset_root)
    records = load_asset_registry(registry_path)
    integrity = validate_materialized_assets(records, asset_root)

    assert summary.record_count == 48
    assert summary.generated_now == 48
    assert summary.reused_existing == 0
    assert integrity.generated_assets == 48
    assert integrity.sha256_verified_assets == 48
    assert (asset_root / records[0].image_reference).is_file()


def test_materialization_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    first_registry = _write_planned_registry(tmp_path / "first")
    second_registry = _write_planned_registry(tmp_path / "second")
    first_assets = tmp_path / "first-assets"
    second_assets = tmp_path / "second-assets"

    materialize_procedural_corpus(first_registry, MATRIX_PATH, first_assets)
    materialize_procedural_corpus(second_registry, MATRIX_PATH, second_assets)
    first_records = load_asset_registry(first_registry)
    second_records = load_asset_registry(second_registry)

    assert [record.image_sha256 for record in first_records] == [
        record.image_sha256 for record in second_records
    ]
    assert (
        hashlib.sha256((first_assets / first_records[0].image_reference).read_bytes()).hexdigest()
        == hashlib.sha256(
            (second_assets / second_records[0].image_reference).read_bytes()
        ).hexdigest()
    )

    repeat = materialize_procedural_corpus(first_registry, MATRIX_PATH, first_assets)
    assert repeat.generated_now == 0
    assert repeat.reused_existing == 48


def test_validate_materialized_assets_rejects_tampered_png(tmp_path: Path) -> None:
    registry_path = _write_planned_registry(tmp_path)
    asset_root = tmp_path / "data"
    materialize_procedural_corpus(registry_path, MATRIX_PATH, asset_root)
    records = load_asset_registry(registry_path)
    first_asset = asset_root / records[0].image_reference
    first_asset.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="not a PNG"):
        validate_materialized_assets(records, asset_root)
