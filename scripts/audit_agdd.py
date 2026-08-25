#!/usr/bin/env python3
"""Audit the pinned official AGDD aircraft glass defect release."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

CLASS_NAMES = {0: "contusion", 1: "scratches", 2: "crack", 3: "spot"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def label_counts(path: Path, expected_values: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != expected_values + 1:
            raise ValueError(f"{path}:{line_number}: expected {expected_values + 1} fields")
        class_id = int(fields[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: unknown class {class_id}")
        coordinates = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        counts[class_id] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    split_reports: dict[str, object] = {}
    all_obb: Counter[int] = Counter()
    all_rect: Counter[int] = Counter()
    manifest_digest = hashlib.sha256()

    for split in ("train", "val"):
        paths = {
            "forward": args.dataset / "data" / "image" / split,
            "backward": args.dataset / "data" / "images" / split,
            "obb": args.dataset / "data" / "labels" / split,
            "rect": args.dataset / "data" / "labels_rect" / split,
        }
        stems = {name: {item.stem for item in path.iterdir()} for name, path in paths.items()}
        reference = stems["forward"]
        if any(values != reference for values in stems.values()):
            raise ValueError(f"Unmatched paired assets in {split}")

        dimensions: Counter[str] = Counter()
        obb_counts: Counter[int] = Counter()
        rect_counts: Counter[int] = Counter()
        identical_pairs = 0
        for stem in sorted(reference):
            forward = paths["forward"] / f"{stem}.png"
            backward = paths["backward"] / f"{stem}.png"
            obb = paths["obb"] / f"{stem}.txt"
            rect = paths["rect"] / f"{stem}.txt"
            forward_size = png_size(forward)
            backward_size = png_size(backward)
            if forward_size != backward_size:
                raise ValueError(f"Pair dimensions differ for {split}/{stem}")
            dimensions[f"{forward_size[0]}x{forward_size[1]}"] += 1
            forward_sha = sha256(forward)
            backward_sha = sha256(backward)
            identical_pairs += forward_sha == backward_sha
            for value in (split, stem, forward_sha, backward_sha, sha256(obb), sha256(rect)):
                manifest_digest.update(value.encode())
            obb_counts.update(label_counts(obb, expected_values=8))
            rect_counts.update(label_counts(rect, expected_values=4))

        if obb_counts != rect_counts:
            raise ValueError(f"OBB and rectangular class counts differ in {split}")
        all_obb.update(obb_counts)
        all_rect.update(rect_counts)
        split_reports[split] = {
            "paired_samples": len(reference),
            "image_files": len(reference) * 2,
            "identical_image_pairs": identical_pairs,
            "dimensions": dict(sorted(dimensions.items())),
            "class_instances": {
                CLASS_NAMES[class_id]: obb_counts[class_id] for class_id in CLASS_NAMES
            },
        }

    report = {
        "dataset": "AGDD",
        "scope": "aircraft glass canopy defect detection with paired illumination",
        "upstream": "https://github.com/core128/AGDD",
        "git_commit": "4b5daa92929934f30b1155033c3ce67b7701960f",
        "license": "CC BY-NC-SA 4.0",
        "class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
        "splits": split_reports,
        "totals": {
            "paired_samples": sum(item["paired_samples"] for item in split_reports.values()),
            "image_files": sum(item["image_files"] for item in split_reports.values()),
            "class_instances": {
                CLASS_NAMES[class_id]: all_obb[class_id] for class_id in CLASS_NAMES
            },
        },
        "checks": {
            "all_pairs_complete": True,
            "all_obb_labels_valid": True,
            "all_rect_labels_valid": True,
            "obb_rect_class_counts_match": all_obb == all_rect,
        },
        "manifest_sha256": manifest_digest.hexdigest(),
        "limitations": [
            "Labels are object-detection boxes, not independent human quality ratings.",
            "The release is noncommercial and share-alike; verify downstream use.",
            "Static paired illumination is not temporal video.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
