"""Release checks for synthetic-to-real aircraft defect transfer."""

from __future__ import annotations

from typing import Any


def assess_transfer_release(
    experiment: dict[str, Any],
    human_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = experiment["aggregate"]
    real = aggregate["real_only"]
    mixed = aggregate["real_plus_synthetic"]
    image_count = experiment["protected_real_validation"]["images"]
    macro_gain = mixed["macro_f1"]["mean"] - real["macro_f1"]["mean"]
    crack_delta = mixed["crack_recall"]["mean"] - real["crack_recall"]["mean"]
    reviewed = None if human_alignment is None else human_alignment.get("case_count")
    agreement = None if human_alignment is None else human_alignment.get("human_agreement")
    checks = {
        "minimum_real_test_images": {
            "value": image_count, "minimum": 200, "passed": image_count >= 200,
        },
        "macro_f1_improvement": {
            "value": macro_gain, "minimum": 0.0, "passed": macro_gain > 0,
        },
        "protected_crack_recall_noninferiority": {
            "value": crack_delta, "minimum": -0.05, "passed": crack_delta >= -0.05,
        },
        "independent_human_review_size": {
            "value": reviewed, "minimum": 100,
            "passed": reviewed is not None and reviewed >= 100,
        },
        "independent_human_agreement": {
            "value": agreement, "minimum": 0.8,
            "passed": agreement is not None and agreement >= 0.8,
        },
    }
    return {
        "schema_version": "1.0",
        "policy": "agdd-transfer-release-v1",
        "decision": "approved" if all(check["passed"] for check in checks.values()) else "blocked",
        "checks": checks,
        "boundary": "research transfer evidence only; never an airworthiness decision",
    }
