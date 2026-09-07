#!/usr/bin/env python3
"""Run a reproducible DLR YOLO baseline and retain a source-free receipt."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    dataset = args.data.resolve()
    if not dataset.is_file():
        raise ValueError(f"Dataset YAML does not exist: {dataset}")
    run = YOLO(args.model).train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        project=str(args.output.parent),
        name=args.output.stem,
        exist_ok=True,
        seed=42,
    )
    run_dir = Path(run.save_dir)
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    final = rows[-1] if rows else {}
    receipt = {
        "schema_version": "1.0",
        "dataset_yaml": dataset.name,
        "model": args.model,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch": args.batch,
        "device_requested": args.device,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "run_directory": str(run_dir),
        "final_epoch_metrics": final,
        "contains_source_images": False,
        "limitations": [
            "Metrics are valid only for the frozen DLR capture-session split.",
            "This detector is not a maintenance or airworthiness decision system.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
