"""Matched real-only / random / evaluator-selected augmentation experiments.

This is an image-level defect classifier, not a bounding-box detector. A frozen
manifest supplies train/validation/test capture groups and synthetic-only scores.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def validate_manifest(rows: list[dict], root: Path) -> dict:
    if not rows:
        raise ValueError("manifest is empty")
    ids, hashes, groups = set(), {}, {}
    train_groups = {r["group"] for r in rows if r["kind"] == "real" and r["split"] == "train"}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("duplicate image ID")
        ids.add(row["id"])
        if row["kind"] not in {"real", "synthetic"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError("invalid kind or split")
        if row["label"] not in (0, 1) or not row["group"]:
            raise ValueError("binary label and capture group required")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("image path escapes data root")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in hashes:
            raise ValueError("duplicate image bytes; deduplicate before partitioning")
        hashes[sha] = row["id"]
        if row["kind"] == "synthetic":
            if row["split"] != "train" or not np.isfinite(row["evaluator_score"]):
                raise ValueError("synthetic data must have a finite score and be train-only")
            if row.get("source_group") is not None and row["source_group"] not in train_groups:
                raise ValueError("synthetic source group must belong to real training data")
        else:
            group = row["group"]
            if group in groups and groups[group] != row["split"]:
                raise ValueError("real capture group crosses partitions")
            groups[group] = row["split"]
    for split in ("train", "val", "test"):
        labels = {r["label"] for r in rows if r["kind"] == "real" and r["split"] == split}
        if labels != {0, 1}:
            raise ValueError(f"{split} requires both real classes")
    return {"image_sha256": hashes, "real_capture_groups": len(groups)}


def run_experiment(
    rows: list[dict], root: Path, *, budget: int, seeds: list[int], recall_margin: float = 0.05
) -> dict:
    audit = validate_manifest(rows, root)
    synthetic = [i for i, row in enumerate(rows) if row["kind"] == "synthetic"]
    if budget < 1 or budget > len(synthetic) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("valid augmentation budget and unique seeds required")
    if not 0 <= recall_margin <= 1:
        raise ValueError("recall margin must be between 0 and 1")
    features = []
    for row in rows:
        with Image.open(root / row["path"]) as image:
            features.append(
                np.asarray(image.convert("L").resize((32, 32)), dtype=float).ravel() / 255
            )
    x = np.asarray(features)
    y = np.asarray([r["label"] for r in rows])
    partitions = {
        s: np.asarray([i for i, r in enumerate(rows) if r["kind"] == "real" and r["split"] == s])
        for s in ("train", "val", "test")
    }
    selected = sorted(synthetic, key=lambda i: (-rows[i]["evaluator_score"], rows[i]["id"]))[
        :budget
    ]
    results = []
    for seed in seeds:
        random = np.random.default_rng(seed).choice(synthetic, budget, replace=False).tolist()
        for arm, extra in [
            ("real_only", []),
            ("random_augmentation", random),
            ("selected_augmentation", selected),
        ]:
            train = np.concatenate([partitions["train"], np.asarray(extra, dtype=int)])
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=1000, random_state=seed
                ),
            )
            model.fit(x[train], y[train])
            validation = model.predict_proba(x[partitions["val"]])[:, 1]
            # Validation-only operating point, equal rule across every arm.
            thresholds = np.arange(0.05, 1, 0.05)
            threshold = max(
                thresholds,
                key=lambda t: (
                    f1_score(y[partitions["val"]], validation >= t, average="macro"),
                    float(t),
                ),
            )
            predictions = model.predict_proba(x[partitions["test"]])[:, 1] >= threshold
            results.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "threshold": float(threshold),
                    "train_rows": len(train),
                    "synthetic_rows": len(extra),
                    "selected_ids": [rows[i]["id"] for i in extra],
                    "macro_f1": float(
                        f1_score(y[partitions["test"]], predictions, average="macro")
                    ),
                    "defect_recall": float(recall_score(y[partitions["test"]], predictions)),
                }
            )
    means = {
        arm: {
            metric: float(np.mean([r[metric] for r in results if r["arm"] == arm]))
            for metric in ("macro_f1", "defect_recall")
        }
        for arm in ("real_only", "random_augmentation", "selected_augmentation")
    }
    selected_mean = means["selected_augmentation"]
    checks = {
        "real_test_size": len(partitions["test"]) >= 200,
        "repeated_seeds": len(seeds) >= 5,
        "beats_real": selected_mean["macro_f1"] > means["real_only"]["macro_f1"],
        "beats_random": selected_mean["macro_f1"] > means["random_augmentation"]["macro_f1"],
        "recall_noninferiority": selected_mean["defect_recall"]
        >= means["real_only"]["defect_recall"] - recall_margin,
    }
    return {
        "schema_version": "1.0",
        "task": "image-level defect presence",
        "status": "descriptive_criteria_met" if all(checks.values()) else "blocked",
        "checks": checks,
        "means": means,
        "runs": results,
        "seeds": seeds,
        "budget": budget,
        "recall_margin": recall_margin,
        "audit": audit,
        "limitations": [
            "Seeds reuse test images; no independent confidence interval claimed.",
            "Manifest timing and untouched-test status require an external experiment record.",
            "This is not independent human evaluator validation or a detector mAP result.",
        ],
    }
