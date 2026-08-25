import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_temporal_qwen.py"
SPEC = importlib.util.spec_from_file_location("run_temporal_qwen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
materialize_frames = MODULE.materialize_frames
percentile = MODULE.percentile
summarize = MODULE.summarize


def test_materialize_frames_produces_requested_ordered_count(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(source)

    frames = materialize_frames([source], 8, tmp_path / "frames")

    assert len(frames) == 8
    assert [path.name for path in frames] == [f"frame-{index:02d}.jpg" for index in range(8)]
    assert all(path.exists() for path in frames)


def test_temporal_summary_reports_accuracy_latency_and_failures() -> None:
    records = [
        {"frame_count": 1, "correct": True, "latency_ms": 10.0, "reference": "clean"},
        {"frame_count": 1, "correct": False, "latency_ms": 20.0, "reference": "crack"},
    ]

    summary = summarize(records)["1"]

    assert summary["accuracy"] == 0.5
    assert summary["latency_p50_ms"] == 15.0
    assert summary["latency_p95_ms"] == 20.0
    assert summary["failure_counts_by_condition"] == {"crack": 1}
    assert percentile([10.0, 20.0], 0.95) == 20.0
