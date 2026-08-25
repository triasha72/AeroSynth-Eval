#!/usr/bin/env python3
"""Run a stronger free Qwen2.5-VL baseline on AeroSynth's development queue."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LABELS = ("no_visible_defect", "corrosion", "surface_crack", "coating_damage")


def normalize(raw: str) -> str | None:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_").rstrip(".,")
    matches = [label for label in LABELS if label in value]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("data/annotations/v0_1_development_annotation_queue.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/qwen25_vl_3b_development.json")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    with args.queue.open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))[: args.limit]

    records = []
    for index, case in enumerate(cases, start=1):
        image_path = (args.data_root / case["image_reference"]).resolve()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": (
                    "Classify this aircraft-panel image. Return exactly one label: "
                    "no_visible_defect, corrosion, surface_crack, or coating_damage."
                )},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        latency_ms = (time.perf_counter() - started) * 1000
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        prediction = normalize(raw)
        records.append({
            "scenario_id": case["scenario_id"], "reference": case["condition"],
            "prediction": prediction, "raw_output": raw, "parse_success": prediction is not None,
            "correct": prediction == case["condition"], "latency_ms": latency_ms,
        })
        print(
            json.dumps({"case": index, "scenario_id": case["scenario_id"], "raw": raw}),
            flush=True,
        )

    payload = {
        "claim_scope": "development_baseline_only", "model_id": MODEL_ID,
        "hardware": platform.platform(), "device": device, "case_count": len(records),
        "parse_rate": sum(row["parse_success"] for row in records) / len(records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "mean_latency_ms": sum(row["latency_ms"] for row in records) / len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
