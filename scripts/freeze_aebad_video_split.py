#!/usr/bin/env python3
"""Freeze deterministic native AeBAD-V windows for temporal VLM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def numeric_frames(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.glob("*.jpg") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def make_cases(
    root: Path, video: str, label: str, numerators: tuple[int, ...], denominator: int
) -> list[dict[str, object]]:
    folder = root / "test" / video / label
    frames = numeric_frames(folder)
    if len(frames) < 16:
        raise ValueError(f"Need at least 16 frames in {folder}")
    cases = []
    for index, numerator in enumerate(numerators):
        center = round(numerator * (len(frames) - 1) / denominator)
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


def build_manifest(root: Path, variant: str = "primary") -> dict[str, object]:
    if variant == "primary":
        numerators, denominator = (1, 2, 3, 4), 5
    elif variant == "confirmatory":
        numerators, denominator = (1, 3, 5, 7, 9), 10
    else:
        raise ValueError(f"Unknown variant: {variant}")
    selection = [
        case
        for label in ("good", "anomaly")
        for case in make_cases(root, "video1", label, numerators, denominator)
    ]
    test = [
        case
        for video in ("video2", "video3")
        for label in ("good", "anomaly")
        for case in make_cases(root, video, label, numerators, denominator)
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
        "variant": variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("primary", "confirmatory"), default="primary")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    payload = build_manifest(args.dataset, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
