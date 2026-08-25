import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eval_agdd_smolvlm2 import selected_labels


def test_selected_labels_respects_manifest(tmp_path: Path) -> None:
    labels = tmp_path / "data" / "labels" / "val"
    labels.mkdir(parents=True)
    for sample_id in ("a", "b", "c"):
        (labels / f"{sample_id}.txt").write_text("0 0 0 0 0\n")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"selection_ids": ["a"], "test_ids": ["b"]}))

    assert [path.stem for path in selected_labels(tmp_path, manifest, "test")] == ["b"]
