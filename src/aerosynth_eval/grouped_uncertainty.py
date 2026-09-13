"""Paired acquisition-group bootstrap; seeds are averaged, not independent data."""

import numpy as np
from sklearn.metrics import f1_score, recall_score


def paired_group_intervals(labels, groups, selected, control, *, draws=1000, seed=2026):
    y = np.asarray(labels)
    groups = np.asarray(groups)
    a, b = np.asarray(selected), np.asarray(control)
    if draws < 100 or a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y):
        raise ValueError("aligned seed-by-image predictions and at least 100 draws required")
    if len(groups) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("aligned groups and both binary classes required")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two independent acquisition groups required")
    indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    values = {"macro_f1_difference": [], "defect_recall_difference": []}
    for _ in range(draws):
        sample = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        if len(np.unique(y[sample])) < 2:
            continue
        for name, metric in [
            ("macro_f1_difference", f1_score),
            ("defect_recall_difference", recall_score),
        ]:
            kw = {"average": "macro", "labels": [0, 1]} if metric is f1_score else {}
            delta = [
                metric(y[sample], x[sample], **kw) - metric(y[sample], z[sample], **kw)
                for x, z in zip(a, b, strict=True)
            ]
            values[name].append(float(np.mean(delta)))
    valid = len(values["macro_f1_difference"])
    return {
        "method": "paired acquisition-group bootstrap; mean across matched seeds",
        "groups": len(unique),
        "requested_draws": draws,
        "valid_draws": valid,
        "status": "estimated" if valid >= 0.9 * draws else "insufficient_valid_draws",
        "intervals_95": {
            k: np.quantile(v, [0.025, 0.975]).tolist() if v else None for k, v in values.items()
        },
        "scope": "conditional on this dataset and fitted models; not external confirmation",
    }
