#!/usr/bin/env python3
"""Measure whether procedural crack images help low-data real AGDD training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.train_agdd_real_baseline import classes_in, image_features


def load_real_images(root: Path, split: str):
    features, labels, groups = [], [], []
    for label_path in sorted((root / "data/labels" / split).glob("*.txt")):
        label = int(2 in classes_in(label_path))
        for directory in ("image", "images"):
            features.append(
                image_features(root / f"data/{directory}/{split}/{label_path.stem}.png")
            )
            labels.append(label)
            groups.append(label_path.stem)
    return np.asarray(features), np.asarray(labels), np.asarray(groups)


def load_synthetic_development(repository: Path):
    features, labels = [], []
    registry = repository / "data/registry/v0_1_asset_registry.jsonl"
    for line in registry.read_text().splitlines():
        record = json.loads(line)
        if record["split"] != "development":
            continue
        features.append(image_features(repository / "data" / record["image_reference"]))
        labels.append(int("-crack-" in record["scenario_id"]))
    return np.asarray(features), np.asarray(labels)


def balanced_sample(features, labels, count: int, generator):
    half = count // 2
    selected = np.concatenate(
        [generator.choice(np.flatnonzero(labels == label), half, replace=False) for label in (0, 1)]
    )
    return features[selected], labels[selected]


def fit_score(train_x, train_y, test_x, test_y, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=seed),
    )
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return {
        "macro_f1": float(f1_score(test_y, prediction, average="macro", zero_division=0)),
        "crack_recall": float(recall_score(test_y, prediction, pos_label=1, zero_division=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("agdd", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    real_x, real_y, _ = load_real_images(args.agdd, "train")
    test_x, test_y, test_groups = load_real_images(args.agdd, "val")
    synthetic_x, synthetic_y = load_synthetic_development(args.repository)
    treatment_rows = min(2 * int(np.sum(synthetic_y == 1)), 2 * int(np.sum(real_y == 1)))
    records = []
    for seed in range(args.seeds):
        generator = np.random.default_rng(seed)
        real_sample_x, real_sample_y = balanced_sample(real_x, real_y, treatment_rows, generator)
        synthetic_sample_x, synthetic_sample_y = balanced_sample(
            synthetic_x, synthetic_y, treatment_rows, generator
        )
        mixed_real_x, mixed_real_y = balanced_sample(real_x, real_y, treatment_rows // 2, generator)
        mixed_synthetic_x, mixed_synthetic_y = balanced_sample(
            synthetic_x, synthetic_y, treatment_rows // 2, generator
        )
        treatments = {
            "real_only": (real_sample_x, real_sample_y),
            "synthetic_only_control": (synthetic_sample_x, synthetic_sample_y),
            "real_plus_synthetic": (
                np.concatenate((mixed_real_x, mixed_synthetic_x)),
                np.concatenate((mixed_real_y, mixed_synthetic_y)),
            ),
        }
        records.append(
            {
                "seed": seed,
                "treatments": {
                    name: fit_score(train_x, train_y, test_x, test_y, seed)
                    for name, (train_x, train_y) in treatments.items()
                },
            }
        )
    aggregate = {
        name: {
            metric: {
                "mean": float(np.mean(values)),
                "sample_standard_deviation": float(np.std(values, ddof=1)),
            }
            for metric in ("macro_f1", "crack_recall")
            for values in [[record["treatments"][name][metric] for record in records]]
        }
        for name in ("real_only", "synthetic_only_control", "real_plus_synthetic")
    }
    result = {
        "schema_version": "1.0",
        "evidence_label": "agdd_real_procedural_transfer_experiment",
        "real_dataset": "AGDD",
        "real_dataset_commit": "4b5daa92929934f30b1155033c3ce67b7701960f",
        "protected_real_validation": {
            "images": len(test_y),
            "paired_cases": len(set(test_groups)),
            "crack_positive_images": int(np.sum(test_y)),
        },
        "equal_training_rows_per_treatment": treatment_rows,
        "seeds": args.seeds,
        "aggregate": aggregate,
        "per_seed": records,
        "limitations": [
            "Only crack labels overlap between the procedural and AGDD taxonomies.",
            "The protected real validation set has only six positive crack images.",
            "Synthetic-only is a transfer control, not real-data model evidence.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
