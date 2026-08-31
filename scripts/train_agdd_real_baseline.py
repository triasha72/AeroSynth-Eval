#!/usr/bin/env python3
"""Train a transparent paired-image multilabel baseline on real AGDD images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["contusion", "scratches", "crack", "spot"]


def classes_in(path):
    return {int(line.split()[0]) for line in Path(path).read_text().splitlines() if line}


def image_features(path):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32).reshape(-1) / 255
    return gray


def load_split(root, split):
    features = []
    labels = []
    for label_path in sorted((root / "data/labels" / split).glob("*.txt")):
        stem = label_path.stem
        features.append(
            np.concatenate(
                [
                    image_features(root / "data/image" / split / f"{stem}.png"),
                    image_features(root / "data/images" / split / f"{stem}.png"),
                ]
            )
        )
        present = classes_in(label_path)
        labels.append([int(index in present) for index in range(4)])
    return np.asarray(features), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train_x, train_y = load_split(args.dataset, "train")
    val_x, val_y = load_split(args.dataset, "val")
    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        ),
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(val_x)
    prediction = (probabilities >= 0.5).astype(int)
    report = classification_report(
        val_y, prediction, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    payload = {
        "schema_version": "1.0",
        "dataset": "AGDD",
        "dataset_license": "CC BY-NC-SA 4.0",
        "model": "paired 32x32 grayscale pixels + one-vs-rest logistic regression",
        "rows": {"train": len(train_y), "validation": len(val_y)},
        "threshold": 0.5,
        "validation": {
            "macro_f1": float(f1_score(val_y, prediction, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(val_y, prediction, average="micro", zero_division=0)),
            "exact_match_accuracy": float(np.mean(np.all(val_y == prediction, axis=1))),
            "classification_report": report,
        },
        "source_commit": "4b5daa92929934f30b1155033c3ce67b7701960f",
        "contains_synthetic_data": False,
        "limitations": [
            "Only 22 validation pairs; uncertainty is high.",
            "Noncommercial license.",
            "Baseline is not an airworthiness or maintenance-release system.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
