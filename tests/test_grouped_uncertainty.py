import numpy as np
import pytest

from aerosynth_eval.grouped_uncertainty import paired_group_intervals


def test_identical_predictions_have_zero_paired_difference():
    y = [0, 1, 0, 1]
    p = np.array([y, y])
    result = paired_group_intervals(y, ["a", "a", "b", "b"], p, p, draws=100)
    assert result["intervals_95"]["macro_f1_difference"] == [0, 0]
    assert result["valid_draws"] == 100


def test_rejects_single_acquisition_group():
    with pytest.raises(ValueError, match="two independent"):
        paired_group_intervals([0, 1], ["a", "a"], [[0, 1]], [[0, 1]], draws=100)
