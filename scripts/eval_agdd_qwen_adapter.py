#!/usr/bin/env python3
"""Evaluate base or QLoRA-adapted Qwen on held-out AGDD validation images."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

CLASS_NAMES = {0: "contusion", 1: "scratches", 2: "crack", 3: "spot"}
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "Inspect this aircraft glass canopy image. Return every visible defect class as a "
    "comma-separated list in numeric order, chosen only from: contusion, scratches, crack, spot."
)


def reference_classes(path: Path) -> list[str]:
    ids = sorted({int(line.split()[0]) for line in path.read_text().splitlines() if line})
    return [CLASS_NAMES[class_id] for class_id in ids]


def predicted_classes(raw: str) -> list[str]:
    lowered = raw.lower()
    return [name for name in CLASS_NAMES.values() if name in lowered]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16, quantization_config=quantization
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28
    )

    records = []
    labels = args.dataset / "data" / "labels" / "val"
    for label_path in sorted(labels.glob("*.txt")):
        image_path = args.dataset / "data" / "image" / "val" / f"{label_path.stem}.png"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path.resolve()}"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        latency_ms = (time.perf_counter() - started) * 1000
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        reference = reference_classes(label_path)
        prediction = predicted_classes(raw)
        records.append(
            {
                "sample_id": label_path.stem,
                "reference": reference,
                "prediction": prediction,
                "raw_output": raw,
                "exact_match": prediction == reference,
                "latency_ms": latency_ms,
            }
        )
        print(json.dumps({"sample": label_path.stem, "raw": raw}), flush=True)

    predicted_distribution = Counter(name for row in records for name in row["prediction"])
    payload = {
        "claim_scope": "agdd_native_held_out_validation",
        "model_id": MODEL_ID,
        "adapter": str(args.adapter) if args.adapter else None,
        "case_count": len(records),
        "exact_match_accuracy": sum(row["exact_match"] for row in records) / len(records),
        "parse_rate": sum(bool(row["prediction"]) for row in records) / len(records),
        "mean_latency_ms": sum(row["latency_ms"] for row in records) / len(records),
        "predicted_class_counts": dict(sorted(predicted_distribution.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
