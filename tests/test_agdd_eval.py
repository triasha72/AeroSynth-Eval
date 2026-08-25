import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_agdd_qwen_adapter.py"
SPEC = importlib.util.spec_from_file_location("eval_agdd_qwen_adapter", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
classification_metrics = MODULE.classification_metrics
predicted_classes = MODULE.predicted_classes


def test_predicted_classes_preserves_canonical_order() -> None:
    assert predicted_classes("SPOT and crack") == ["crack", "spot"]


def test_classification_metrics_reports_macro_and_per_class_scores() -> None:
    records = [
        {"reference": ["contusion"], "prediction": ["contusion", "spot"]},
        {"reference": ["spot"], "prediction": ["spot"]},
    ]

    metrics = classification_metrics(records)

    assert metrics["per_class"]["contusion"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1,
    }
    assert metrics["per_class"]["spot"]["precision"] == 0.5
    assert metrics["per_class"]["spot"]["recall"] == 1.0
    assert metrics["micro_f1"] == 0.8
