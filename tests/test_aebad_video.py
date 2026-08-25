import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eval_aebad_video_smolvlm2 import parse_prediction, sampled_frames
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
            for index in range(20):
                (folder / f"{index}.jpg").write_bytes(str(index).encode())
            (folder / "._0.jpg").write_bytes(b"resource fork")
    payload = build_manifest(tmp_path)
    assert {case["video"] for case in payload["selection_cases"]} == {"video1"}
    assert {case["video"] for case in payload["test_cases"]} == {"video2", "video3"}
