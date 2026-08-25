import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eval_aebad_video_smolvlm2 import parse_prediction, sampled_frames
from fit_aebad_video_thresholds import fit
from freeze_aebad_video_split import build_manifest


def test_sampling_has_requested_size_and_endpoints() -> None:
    frames = [str(index) for index in range(16)]
    assert sampled_frames(frames, 1) == ["8"]
    assert sampled_frames(frames, 4) == ["0", "5", "10", "15"]


def test_parser_is_bounded() -> None:
    assert parse_prediction("Anomaly") == "anomaly"
    assert parse_prediction("normal blade") == "good"
    assert parse_prediction("uncertain") is None


def test_manifest_keeps_video1_out_of_test(tmp_path: Path) -> None:
    for video in ("video1", "video2", "video3"):
        for label in ("good", "anomaly"):
            folder = tmp_path / "test" / video / label
            folder.mkdir(parents=True)
            for index in range(200):
                (folder / f"{index}.jpg").write_bytes(str(index).encode())
            (folder / "._0.jpg").write_bytes(b"resource fork")
    payload = build_manifest(tmp_path)
    assert {case["video"] for case in payload["selection_cases"]} == {"video1"}
    assert {case["video"] for case in payload["test_cases"]} == {"video2", "video3"}
    confirmatory = build_manifest(tmp_path, "confirmatory")
    primary_frames = {frame for case in payload["test_cases"] for frame in case["frames"]}
    confirmatory_frames = {
        frame for case in confirmatory["test_cases"] for frame in case["frames"]
    }
    assert primary_frames.isdisjoint(confirmatory_frames)


def test_threshold_fit_uses_selection_labels() -> None:
    records = [
        {"frame_count": 1, "reference": "good", "anomaly_probability": 0.1},
        {"frame_count": 1, "reference": "good", "anomaly_probability": 0.2},
        {"frame_count": 1, "reference": "anomaly", "anomaly_probability": 0.7},
        {"frame_count": 1, "reference": "anomaly", "anomaly_probability": 0.8},
    ]
    result = fit(records)["thresholds"]["1"]
    assert result["selection_balanced_accuracy"] == 1.0
