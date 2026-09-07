import pytest

from aerosynth_eval.downstream_utility import assess_downstream_utility


def test_downstream_utility_passes_ordered_real_results() -> None:
    result = assess_downstream_utility(
        {
            "untouched_real_test_images": 240,
            "real_test_used_for_selection": False,
            "selection_experiments": [
                {"mean_evaluator_score": score, "real_macro_f1_gain": gain}
                for score, gain in zip(
                    [0.2, 0.4, 0.6, 0.8, 0.9],
                    [0.001, 0.008, 0.012, 0.018, 0.025],
                    strict=True,
                )
            ],
        }
    )
    assert result["decision"] == "supported"
    assert result["checks"]["evaluator_score_downstream_correlation"]["value"] == pytest.approx(1.0)


def test_downstream_utility_blocks_small_or_leaked_test() -> None:
    result = assess_downstream_utility(
        {
            "untouched_real_test_images": 44,
            "real_test_used_for_selection": True,
            "selection_experiments": [{"mean_evaluator_score": 0.5, "real_macro_f1_gain": 0.01}],
        }
    )
    assert result["decision"] == "blocked"
    assert not result["checks"]["untouched_real_test_size"]["passed"]
    assert not result["checks"]["no_real_test_selection_leakage"]["passed"]
