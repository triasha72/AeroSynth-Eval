import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_streaming_qwen.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_streaming_qwen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BoundedDropOldestQueue = MODULE.BoundedDropOldestQueue
summarize_mode = MODULE.summarize_mode


def test_bounded_queue_drops_oldest_and_recovers() -> None:
    queue = BoundedDropOldestQueue(2)
    queue.put("one")
    queue.put("two")
    queue.put("three")

    assert queue.dropped == 1
    assert queue.get() == "two"
    assert queue.get() == "three"


def test_streaming_summary_reports_latency_throughput_and_failures() -> None:
    records = [
        {"correct": True, "latency_ms": 10.0, "reference": "clean"},
        {"correct": False, "latency_ms": 30.0, "reference": "crack"},
    ]

    summary = summarize_mode(records, wall_seconds=2.0)

    assert summary["accuracy"] == 0.5
    assert summary["latency_p50_ms"] == 20.0
    assert summary["latency_p95_ms"] == 30.0
    assert summary["throughput_windows_per_second"] == 1.0
    assert summary["failure_counts_by_condition"] == {"crack": 1}
