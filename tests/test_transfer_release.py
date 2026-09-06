import json
from pathlib import Path

from aerosynth_eval.transfer_release import assess_transfer_release


def test_published_transfer_is_blocked_by_crack_recall_and_sample_size():
    experiment = json.loads(
        (Path(__file__).parents[1] / "reports/agdd_transfer_experiment_v1.json").read_text()
    )
    result = assess_transfer_release(experiment)
    assert result["decision"] == "blocked"
    assert result["checks"]["macro_f1_improvement"]["passed"]
    assert not result["checks"]["protected_crack_recall_noninferiority"]["passed"]
    assert not result["checks"]["minimum_real_test_images"]["passed"]
