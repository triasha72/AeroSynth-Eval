#!/usr/bin/env python3
"""Freeze deterministic native AeBAD-V windows for temporal VLM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def numeric_frames(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.jpg"), key=lambda path: int(path.stem))


def make_cases(root: Path, video: str, label: str, windows: int = 4) -> list[dict[str, object]]:
    folder = root / "test" / video / label
    frames = numeric_frames(folder)
    if len(frames) < 16:
        raise ValueError(f"Need at least 16 frames in {folder}")
    cases = []
    for index in range(windows):
        center = round((index + 1) * (len(frames) - 1) / (windows + 1))
        start = max(0, min(len(frames) - 16, center - 8))
        selected = frames[start : start + 16]
        relative = [str(path.relative_to(root)) for path in selected]
        cases.append({
            "case_id": f"{video}-{label}-{index + 1}",
            "video": video,
            "label": label,
            "frames": relative,
            "frame_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in selected],
        })
    return cases


def build_manifest(root: Path) -> dict[str, object]:
    selection = [
        case for label in ("good", "anomaly") for case in make_cases(root, "video1", label)
    ]
    test = [
        case
        for video in ("video2", "video3")
        for label in ("good", "anomaly")
        for case in make_cases(root, video, label)
    ]
    return {
        "dataset": "AeBAD-V",
        "source": "https://github.com/zhangzilongc/MMR",
        "license": "CC BY 4.0",
        "window_frames": 16,
        "frame_count_ablation": [1, 4, 8, 16],
        "selection_cases": selection,
        "test_cases": test,
        "policy": "video1 selection; video2 and video3 protected test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    payload = build_manifest(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
