import numpy as np
import pytest
from PIL import Image

from aerosynth_eval.utility_experiment import run_experiment, validate_manifest


def sample(tmp_path):
    rows = []
    rng = np.random.default_rng(12)
    for split in ("train", "val", "test"):
        for label in (0, 1):
            for n in range(3):
                name = f"{split}-{label}-{n}"
                pixels = rng.integers(label * 100, label * 100 + 80, (32, 32), dtype=np.uint8)
                Image.fromarray(pixels).save(tmp_path / f"{name}.png")
                rows.append(
                    {
                        "id": name,
                        "path": f"{name}.png",
                        "group": name,
                        "split": split,
                        "kind": "real",
                        "label": label,
                    }
                )
    for i in range(4):
        name = f"synthetic-{i}"
        Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8)).save(
            tmp_path / f"{name}.png"
        )
        rows.append(
            {
                "id": name,
                "path": f"{name}.png",
                "group": name,
                "split": "train",
                "kind": "synthetic",
                "label": i % 2,
                "evaluator_score": float(i),
            }
        )
    return rows


def test_matched_budget_and_small_test_block(tmp_path):
    result = run_experiment(sample(tmp_path), tmp_path, budget=2, seeds=[17, 29])
    assert result["status"] == "blocked"
    assert {r["synthetic_rows"] for r in result["runs"] if r["arm"] != "real_only"} == {2}
    assert len(result["runs"]) == 6


def test_group_leakage_rejected(tmp_path):
    rows = sample(tmp_path)
    rows[6]["group"] = rows[0]["group"]
    with pytest.raises(ValueError, match="crosses"):
        validate_manifest(rows, tmp_path)


def test_synthetic_test_derivation_rejected(tmp_path):
    rows = sample(tmp_path)
    rows[-1]["source_group"] = rows[12]["group"]
    with pytest.raises(ValueError, match="source group"):
        validate_manifest(rows, tmp_path)
