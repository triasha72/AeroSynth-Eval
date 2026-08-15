import json
from pathlib import Path

import pytest

from aerosynth_eval.judge_registry import load_judge_specs


def test_registry_requires_multiple_families(tmp_path: Path) -> None:
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            [
                {"judge_id": "qwen", "family": "qwen2_vl", "model_id": "m1"},
                {"judge_id": "smol", "family": "smolvlm", "model_id": "m2"},
            ]
        )
    )
    assert len(load_judge_specs(path)) == 2


def test_registry_rejects_single_family(tmp_path: Path) -> None:
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            [
                {"judge_id": "qwen1", "family": "qwen", "model_id": "m1"},
                {"judge_id": "qwen2", "family": "qwen", "model_id": "m2"},
            ]
        )
    )
    with pytest.raises(ValueError):
        load_judge_specs(path)
