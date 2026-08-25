#!/usr/bin/env python3
"""Create chat-style VLM manifests from the pinned AGDD release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NAMES = {0: "contusion", 1: "scratches", 2: "crack", 3: "spot"}


def classes_in(label_path: Path) -> list[str]:
    class_ids = {int(line.split()[0]) for line in label_path.read_text().splitlines() if line}
    return [CLASS_NAMES[class_id] for class_id in sorted(class_ids)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    for split in ("train", "val"):
        records = []
        labels = args.dataset / "data" / "labels" / split
        for label_path in sorted(labels.glob("*.txt")):
            stem = label_path.stem
            answer = ", ".join(classes_in(label_path))
            records.append(
                {
                    "id": f"agdd-{split}-{stem}",
                    "images": [
                        str((args.dataset / "data" / "image" / split / f"{stem}.png").resolve()),
                        str((args.dataset / "data" / "images" / split / f"{stem}.png").resolve()),
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Inspect the paired forward- and backward-illumination aircraft "
                                "canopy images. Return the visible defect classes as a "
                                "comma-separated list chosen from: contusion, scratches, "
                                "crack, spot."
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    "source": "AGDD",
                    "license": "CC BY-NC-SA 4.0",
                }
            )
        destination = args.output / f"{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
        )
        summary[split] = len(records)

    (args.output / "MANIFEST.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/core128/AGDD",
                "source_commit": "4b5daa92929934f30b1155033c3ce67b7701960f",
                "license": "CC BY-NC-SA 4.0",
                "task": "paired-illumination multilabel classification",
                "splits": summary,
                "warning": (
                    "Noncommercial share-alike dataset; do not mix its validation split "
                    "into training."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
