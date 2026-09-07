#!/usr/bin/env python3
"""Train a small real-data dent-presence baseline from the DLR release ZIP."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aerosynth_eval.dlr_dent_detection import (
    DlrDentRecord,
    audit_capture_leakage,
    scan_archive,
    session_groups,
)


def _features(archive: ZipFile, records: list[DlrDentRecord]) -> np.ndarray:
    values: list[np.ndarray] = []
    for record in records:
        with Image.open(BytesIO(archive.read(record.image_member))) as image:
            pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32).reshape(-1)
        values.append(pixels / 255.0)
    return np.asarray(values)


def _balanced_holdout(
    records: list[DlrDentRecord], gap_seconds: int, fraction: float, seed: int
) -> tuple[list[DlrDentRecord], list[DlrDentRecord]]:
    """Choose a group-held-out partition close in size and dent prevalence.

    GroupShuffleSplit keeps sessions intact but does not stratify labels.  The
    public release has only 43 inferred sessions, so several deterministic
    candidates are evaluated by aggregate label counts instead of leaking
    individual images between partitions.
    """

    groups = session_groups(records, gap_seconds)
    group_values = np.asarray([groups[record.image_member] for record in records])
    labels = np.asarray([int(record.has_dent) for record in records])
    indexes = np.arange(len(records))
    overall_rate = float(np.mean(labels))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(n_splits=512, test_size=fraction, random_state=seed)
    for train, held_out in splitter.split(indexes, groups=group_values):
        held_labels = labels[held_out]
        if len(np.unique(held_labels)) != 2 or len(np.unique(labels[train])) != 2:
            continue
        size_error = abs((len(held_out) / len(records)) - fraction)
        prevalence_error = abs(float(np.mean(held_labels)) - overall_rate)
        score = size_error + 2 * prevalence_error
        if best is None or score < best[0]:
            best = (score, train, held_out)
    if best is None:
        raise ValueError("Could not create a class-balanced capture-session split")
    return ([records[index] for index in best[1]], [records[index] for index in best[2]])


def _split_records(
    records: list[DlrDentRecord], gap_seconds: int, seed: int
) -> dict[str, list[DlrDentRecord]]:
    train_val, test = _balanced_holdout(records, gap_seconds, 0.20, seed)
    train, validation = _balanced_holdout(train_val, gap_seconds, 0.1875, seed + 1)
    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def _limit(records: list[DlrDentRecord], limit: int, seed: int) -> list[DlrDentRecord]:
    if limit <= 0 or len(records) <= limit:
        return records
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(records), size=limit, replace=False))
    return [records[index] for index in selected]


def _choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        prediction = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, prediction, average="binary", zero_division=0
        )
        candidates.append((recall >= 0.90, precision, f1, float(threshold)))
    qualifying = [candidate for candidate in candidates if candidate[0]]
    return max(qualifying or candidates, key=lambda candidate: (candidate[1], candidate[2]))[3]


def _metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | list[list[int]]]:
    prediction = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    result: dict[str, float | list[list[int]]] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(np.mean(y_true)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-gap-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    records = scan_archive(str(args.archive))
    partitions = _split_records(records, args.group_gap_seconds, args.seed)
    for index, name in enumerate(("train", "validation", "test")):
        partitions[name] = _limit(partitions[name], getattr(args, f"max_{name}"), args.seed + index)
    if min(len(partitions[name]) for name in partitions) == 0:
        raise ValueError("Capture-session split produced an empty partition; lower the session gap")

    with ZipFile(args.archive) as archive:
        feature_sets = {name: _features(archive, rows) for name, rows in partitions.items()}
    target_sets = {
        name: np.asarray([int(record.has_dent) for record in rows])
        for name, rows in partitions.items()
    }
    if len(np.unique(target_sets["train"])) != 2:
        raise ValueError("Training split has only one class; use a different seed or session gap")
    classifier = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1000, random_state=args.seed
    )
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(feature_sets["train"], target_sets["train"])
    validation_probabilities = model.predict_proba(feature_sets["validation"])[:, 1]
    threshold = _choose_threshold(target_sets["validation"], validation_probabilities)
    test_probabilities = model.predict_proba(feature_sets["test"])[:, 1]
    leakage = audit_capture_leakage(records, args.group_gap_seconds)
    payload = {
        "schema_version": "1.0",
        "dataset": "DLR aircraft-dent dataset (Zenodo 10.5281/zenodo.17900121)",
        "dataset_license": "MIT",
        "task": "dent-presence triage from inspection images",
        "model": "32x32 grayscale pixels + balanced logistic regression",
        "contains_synthetic_data": False,
        "source_split_policy": (
            "release folders were not used for scoring; records were re-split by "
            "chronological capture session"
        ),
        "capture_session_leakage_in_release": leakage,
        "group_gap_seconds": args.group_gap_seconds,
        "rows": {name: len(rows) for name, rows in partitions.items()},
        "positive_rows": {name: int(target_sets[name].sum()) for name in partitions},
        "threshold_selected_on_validation": threshold,
        "validation": _metrics(target_sets["validation"], validation_probabilities, threshold),
        "test": _metrics(target_sets["test"], test_probabilities, threshold),
        "limitations": [
            "This is a dent-presence triage baseline, not a bounding-box detector or "
            "maintenance decision system.",
            "DLR labels are optical-tracking-derived; they are not independent human "
            "judgments of generated images.",
            "The session boundary is inferred from timestamp gaps and should be reviewed "
            "against acquisition logs before deployment claims.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": payload["rows"], "test": payload["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
