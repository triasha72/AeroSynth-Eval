from pathlib import Path

import pytest

from aerosynth_eval.contracts import AnnotationStatus, DatasetSplit, ImageProvenance
from aerosynth_eval.dataset import load_manifest, summarize_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "examples" / "v0_1_manifest.jsonl"


def test_load_manifest_reads_versioned_fixture() -> None:
    records = load_manifest(MANIFEST_PATH)

    assert len(records) == 6
    assert records[0].example_id == "fuselage-corrosion-001"
    assert records[-1].split is DatasetSplit.TEST


def test_summarize_manifest_counts_expected_categories() -> None:
    summary = summarize_manifest(load_manifest(MANIFEST_PATH))

    assert summary.splits[DatasetSplit.DEVELOPMENT] == 4
    assert summary.splits[DatasetSplit.TEST] == 2
    assert summary.provenance[ImageProvenance.SYNTHETIC] == 6
    assert summary.annotation_status[AnnotationStatus.UNLABELED] == 6


def test_load_manifest_rejects_duplicate_example_ids(tmp_path: Path) -> None:
    first_record = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()[0]
    duplicate_manifest = tmp_path / "duplicate.jsonl"
    duplicate_manifest.write_text(f"{first_record}\n{first_record}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate example_id"):
        load_manifest(duplicate_manifest)
