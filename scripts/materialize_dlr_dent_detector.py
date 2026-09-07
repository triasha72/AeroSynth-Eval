#!/usr/bin/env python3
"""Create a local, session-held-out YOLO dataset from the DLR release ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from aerosynth_eval.dlr_dent_detection import DlrDentRecord, scan_archive, session_groups


def _holdout(
    records: list[DlrDentRecord], gap_seconds: int, fraction: float, seed: int
) -> tuple[list[DlrDentRecord], list[DlrDentRecord]]:
    groups = session_groups(records, gap_seconds)
    values = np.asarray([groups[record.image_member] for record in records])
    indexes = np.arange(len(records))
    labels = np.asarray([record.has_dent for record in records])
    expected_rate = float(np.mean(labels))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(n_splits=512, test_size=fraction, random_state=seed)
    for kept, held_out in splitter.split(indexes, groups=values):
        held_labels = labels[held_out]
        if len(np.unique(held_labels)) != 2 or len(np.unique(labels[kept])) != 2:
            continue
        score = abs(len(held_out) / len(records) - fraction) + 2 * abs(
            float(np.mean(held_labels)) - expected_rate
        )
        if best is None or score < best[0]:
            best = (score, kept, held_out)
    if best is None:
        raise ValueError("Could not create a class-balanced session holdout")
    kept, held_out = best[1:]
    return ([records[index] for index in kept], [records[index] for index in held_out])


def partitions(
    records: list[DlrDentRecord], gap_seconds: int, seed: int
) -> dict[str, list[DlrDentRecord]]:
    """Create a deterministic session-separated train/validation/test partition."""

    train_validation, test = _holdout(records, gap_seconds, 0.20, seed)
    train, validation = _holdout(train_validation, gap_seconds, 0.1875, seed + 1)
    return {"train": train, "val": validation, "test": test}


def materialize(archive_path: Path, output: Path, gap_seconds: int, seed: int) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = partitions(scan_archive(str(archive_path)), gap_seconds, seed)
    with ZipFile(archive_path) as archive:
        for split, records in rows.items():
            for directory in ("images", "labels", "masks"):
                (output / directory / split).mkdir(parents=True, exist_ok=True)
            for record in records:
                target = output / "images" / split / f"{record.image_id}.png"
                target.write_bytes(archive.read(record.image_member))
                if record.mask_member is not None:
                    (output / "masks" / split / f"{record.image_id}.png").write_bytes(
                        archive.read(record.mask_member)
                    )
                label = output / "labels" / split / f"{record.image_id}.txt"
                label.write_bytes(archive.read(record.label_member) if record.label_member else b"")
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnames: [dent]\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "1.0",
        "source_archive": archive_path.name,
        "source_data_committed": False,
        "split_policy": "capture-session group split",
        "group_gap_seconds": gap_seconds,
        "seed": seed,
        "rows": {name: len(records) for name, records in rows.items()},
        "positive_rows": {
            name: sum(record.has_dent for record in records) for name, records in rows.items()
        },
        "next_step": "Train a detector using dataset.yaml; keep the test partition untouched.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--group-gap-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    receipt = materialize(args.archive, args.output, args.group_gap_seconds, args.seed)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
