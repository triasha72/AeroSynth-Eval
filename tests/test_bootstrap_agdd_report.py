import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "bootstrap_agdd_report.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_agdd_report", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
bootstrap_intervals = MODULE.bootstrap_intervals


def test_bootstrap_intervals_are_deterministic_and_bounded() -> None:
    records = [
        {"exact_match": True, "latency_ms": 10.0},
        {"exact_match": False, "latency_ms": 30.0},
    ]

    first = bootstrap_intervals(records, samples=100, seed=3)
    second = bootstrap_intervals(records, samples=100, seed=3)

    assert first == second
    assert first["exact_match_accuracy"] == {"low": 0.0, "high": 1.0}
    assert first["mean_latency_ms"] == {"low": 10.0, "high": 30.0}
