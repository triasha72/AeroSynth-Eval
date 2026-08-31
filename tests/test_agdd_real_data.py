from pathlib import Path

from scripts.train_agdd_real_baseline import classes_in


def test_classes_in_reads_multilabel_annotations(tmp_path: Path):
    path = tmp_path / "label.txt"
    path.write_text("2 0.1 0.2 0.3 0.4\n0 0.2 0.3 0.4 0.5\n")
    assert classes_in(path) == {0, 2}
