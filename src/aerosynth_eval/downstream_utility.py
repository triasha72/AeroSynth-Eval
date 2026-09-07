"""Validate a synthetic-image evaluator through untouched real-image performance."""

from __future__ import annotations

from typing import Any

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including ties, without an extra SciPy dependency."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def assess_downstream_utility(artifact: dict[str, Any]) -> dict[str, Any]:
    """Check whether evaluator-based selection predicts real detector improvement."""

    experiments = artifact.get("selection_experiments", [])
    scores = np.asarray([row["mean_evaluator_score"] for row in experiments], dtype=float)
    gains = np.asarray([row["real_macro_f1_gain"] for row in experiments], dtype=float)
    correlation = None
    if len(experiments) >= 2 and len(set(scores)) > 1 and len(set(gains)) > 1:
        correlation = float(np.corrcoef(_ranks(scores), _ranks(gains))[0, 1])
    real_images = artifact.get("untouched_real_test_images")
    leakage = artifact.get("real_test_used_for_selection")
    checks: dict[str, dict[str, Any]] = {
        "minimum_selection_experiments": {
            "value": len(experiments),
            "minimum": 5,
            "passed": len(experiments) >= 5,
        },
        "untouched_real_test_size": {
            "value": real_images,
            "minimum": 200,
            "passed": real_images is not None and real_images >= 200,
        },
        "no_real_test_selection_leakage": {
            "value": leakage,
            "required": False,
            "passed": leakage is False,
        },
        "evaluator_score_downstream_correlation": {
            "value": correlation,
            "minimum": 0.5,
            "passed": correlation is not None and correlation >= 0.5,
        },
        "positive_mean_real_macro_f1_gain": {
            "value": None if not len(gains) else float(np.mean(gains)),
            "minimum": 0.0,
            "passed": bool(len(gains)) and float(np.mean(gains)) > 0.0,
        },
    }
    return {
        "schema_version": "1.0",
        "policy": "aerosynth-downstream-utility-v1",
        "decision": "supported" if all(check["passed"] for check in checks.values()) else "blocked",
        "checks": checks,
        "claim_scope": "synthetic-image selection utility on an untouched real-image benchmark",
        "non_claims": [
            "Dataset labels are not new independent reviews of generated images.",
            "This evidence is not an airworthiness or maintenance decision.",
        ],
    }
