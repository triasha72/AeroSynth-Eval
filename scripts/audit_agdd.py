#!/usr/bin/env python3
"""Audit the pinned official AGDD aircraft glass-defect release."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

CLASS_NAMES = {0: "contusion", 1: "scratches", 2: "crack", 3: "spot"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path):
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def label_counts(path, values):
    counts = Counter()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != values + 1:
            raise ValueError(f"{path}:{number}: invalid field count")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id not in CLASS_NAMES or not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(f"{path}:{number}: invalid annotation")
        counts[class_id] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    splits = {}
    totals = Counter()
    manifest = hashlib.sha256()
    for split in ("train", "val"):
        paths = {
            "forward": args.dataset / "data/image" / split,
            "backward": args.dataset / "data/images" / split,
            "obb": args.dataset / "data/labels" / split,
            "rect": args.dataset / "data/labels_rect" / split,
        }
        stems = {name: {item.stem for item in path.iterdir()} for name, path in paths.items()}
        if any(values != stems["forward"] for values in stems.values()):
            raise ValueError(f"Unmatched assets in {split}")
        counts = Counter()
        dimensions = Counter()
        identical = 0
        for stem in sorted(stems["forward"]):
            forward = paths["forward"] / f"{stem}.png"
            backward = paths["backward"] / f"{stem}.png"
            if png_size(forward) != png_size(backward):
                raise ValueError(f"Dimension mismatch: {stem}")
            dimensions[f"{png_size(forward)[0]}x{png_size(forward)[1]}"] += 1
            forward_sha = sha256(forward)
            backward_sha = sha256(backward)
            identical += forward_sha == backward_sha
            obb_path = paths["obb"] / f"{stem}.txt"
            rect_path = paths["rect"] / f"{stem}.txt"
            obb = label_counts(obb_path, 8)
            rect = label_counts(rect_path, 4)
            if obb != rect:
                raise ValueError(f"Annotation mismatch: {stem}")
            counts.update(obb)
            for value in (
                split,
                stem,
                forward_sha,
                backward_sha,
                sha256(obb_path),
                sha256(rect_path),
            ):
                manifest.update(value.encode())
        totals.update(counts)
        splits[split] = {
            "paired_samples": len(stems["forward"]),
            "image_files": 2 * len(stems["forward"]),
            "identical_image_pairs": identical,
            "dimensions": dict(dimensions),
            "class_instances": {CLASS_NAMES[key]: counts[key] for key in CLASS_NAMES},
        }
    report = {
        "dataset": "AGDD",
        "upstream": "https://github.com/core128/AGDD",
        "git_commit": "4b5daa92929934f30b1155033c3ce67b7701960f",
        "license": "CC BY-NC-SA 4.0",
        "splits": splits,
        "totals": {
            "paired_samples": sum(item["paired_samples"] for item in splits.values()),
            "class_instances": {CLASS_NAMES[key]: totals[key] for key in CLASS_NAMES},
        },
        "manifest_sha256": manifest.hexdigest(),
        "checks": {"all_pairs_complete": True, "annotations_valid": True},
        "limitations": [
            "Noncommercial share-alike dataset.",
            "Labels are detection boxes, not independent quality ratings.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
