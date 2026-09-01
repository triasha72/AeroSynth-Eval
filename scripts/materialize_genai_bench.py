#!/usr/bin/env python3
"""Materialize the pinned GenAI-Bench human-vote parquet for local evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from aerosynth_eval.preference_benchmark import (
    PreferenceTask,
    normalize_record,
    summarize_manifest,
    write_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def image_suffix(image: dict[str, object]) -> str:
    suffix = Path(str(image.get("path") or "")).suffix.casefold()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    image_directory = args.output_directory / "images"
    image_directory.mkdir(exist_ok=True)
    records = []
    for index, row in enumerate(table.to_pylist()):
        paths = []
        for side in ("left", "right"):
            image = row[f"{side}_image"]
            image_bytes = image.get("bytes")
            if not image_bytes:
                raise ValueError(f"Row {index} has no embedded {side} image")
            path = image_directory / f"{index:05d}_{side}{image_suffix(image)}"
            path.write_bytes(image_bytes)
            paths.append(path)
        records.append(
            normalize_record(
                row,
                task=PreferenceTask.IMAGE_GENERATION,
                source_index=index,
                left_image_path=paths[0],
                right_image_path=paths[1],
            )
        )

    manifest_path = write_manifest(records, args.output_directory / "preferences.jsonl")
    summary = summarize_manifest(records).model_dump(mode="json")
    payload = {
        "schema_version": "1.0",
        "dataset_id": "TIGER-Lab/GenAI-Bench",
        "source_split": "image_generation/test_v1",
        "dataset_license": "CC-BY-4.0",
        "annotation_type": "human_pairwise_preference_votes",
        "source_sha256": sha256(args.parquet),
        "manifest_sha256": sha256(manifest_path),
        **summary,
        "contains_prompts_or_images": False,
        "limitations": [
            "Human preferences concern generated-image quality, not aircraft "
            "inspection correctness.",
            "The deterministic partitions support evaluator development but do "
            "not create new human votes.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
