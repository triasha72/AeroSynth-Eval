"""Convert official RichHF-18K TFRecords to normalized score-only JSONL.

The official RichHF repository contains labels and filenames, not the images.
This utility is intentionally optional: install TensorFlow separately if you
choose to parse the upstream TFRecords locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tfrecord", type=Path)
parser.add_argument("--split", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ModuleNotFoundError as error:
    raise SystemExit(
        
            "TensorFlow is required only for this converter. "
            "Install a Python-3.12-compatible TensorFlow build "
            "in a separate environment if needed."
        
    ) from error
feature_description = {
    "filename": tf.io.FixedLenFeature([], tf.string),
    "aesthetics_score": tf.io.FixedLenFeature([], tf.float32),
    "artifact_score": tf.io.FixedLenFeature([], tf.float32),
    "misalignment_score": tf.io.FixedLenFeature([], tf.float32),
    "overall_score": tf.io.FixedLenFeature([], tf.float32),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as stream:
    for index, raw in enumerate(tf.data.TFRecordDataset([str(args.tfrecord)])):
        item = tf.io.parse_single_example(raw, feature_description)
        row = {
            "example_id": f"richhf-{args.split}-{index:06d}",
            "split": args.split,
            "filename": item["filename"].numpy().decode(),
            "aesthetics_score": float(item["aesthetics_score"].numpy()),
            "artifact_score": float(item["artifact_score"].numpy()),
            "misalignment_score": float(item["misalignment_score"].numpy()),
            "overall_score": float(item["overall_score"].numpy()),
        }
        stream.write(json.dumps(row, sort_keys=True) + "\n")
print(args.output)
