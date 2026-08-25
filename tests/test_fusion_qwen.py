import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_fusion_qwen.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_fusion_qwen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
noisy_transcript = MODULE.noisy_transcript
summarize = MODULE.summarize


def test_noisy_transcript_is_deterministic_and_preserves_some_words() -> None:
    source = "one two three four five six"
    assert noisy_transcript(source) == "one [inaudible] three four [inaudible] six"


def test_fusion_summary_slices_modes_and_failures() -> None:
    records = [
        {"mode": "fusion", "correct": True, "latency_ms": 10.0, "reference": "clean"},
        {"mode": "fusion", "correct": False, "latency_ms": 30.0, "reference": "crack"},
        {"mode": "vision_only", "correct": True, "latency_ms": 20.0, "reference": "clean"},
    ]

    summary = summarize(records)

    assert summary["fusion"]["accuracy"] == 0.5
    assert summary["fusion"]["latency_p50_ms"] == 20.0
    assert summary["fusion"]["failure_counts_by_condition"] == {"crack": 1}
    assert summary["vision_only"]["accuracy"] == 1.0
