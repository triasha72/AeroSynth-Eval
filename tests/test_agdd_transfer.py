import json
from pathlib import Path

import numpy as np

from scripts.run_agdd_transfer_experiment import balanced_sample


def test_transfer_treatments_use_balanced_equal_size_samples():
    features = np.arange(40).reshape(20, 2)
    labels = np.asarray([0] * 10 + [1] * 10)
    selected_x, selected_y = balanced_sample(features, labels, 8, np.random.default_rng(42))
    assert len(selected_x) == 8
    np.testing.assert_array_equal(np.bincount(selected_y), [4, 4])


def test_published_transfer_result_uses_protected_real_validation():
    root = Path(__file__).parents[1]
    result = json.loads((root / "reports/agdd_transfer_experiment_v1.json").read_text())
    assert result["real_dataset"] == "AGDD"
    assert result["protected_real_validation"]["images"] == 44
    assert result["protected_real_validation"]["crack_positive_images"] == 6
    assert result["seeds"] == 10
