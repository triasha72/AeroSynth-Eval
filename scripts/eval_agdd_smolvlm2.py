#!/usr/bin/env python3
"""Evaluate the independent SmolVLM2 model family on a frozen AGDD subset."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from eval_agdd_qwen_adapter import (
    PROMPT,
    calibration_metrics,
    classification_metrics,
    predicted_classes,
    reference_classes,
)

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def selected_labels(dataset: Path, manifest: Path, subset: str) -> list[Path]:
    payload = json.loads(manifest.read_text())
    allowed = set(payload[f"{subset}_ids"])
    labels = dataset / "data" / "labels" / "val"
    return [path for path in sorted(labels.glob("*.txt")) if path.stem in allowed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("selection", "test"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    import torch
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    ).eval()
    records = []
    for label_path in selected_labels(args.dataset, args.split_manifest, args.subset):
        image_path = args.dataset / "data" / "image" / "val" / f"{label_path.stem}.png"
        messages = [{"role": "user", "content": [
            {"type": "image", "url": f"file://{image_path.resolve()}"},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_length = inputs["input_ids"].shape[-1]
        tokens = generated.sequences[:, prompt_length:]
        log_probs = [
            score.log_softmax(dim=-1).gather(1, tokens[:, index, None]).squeeze(1)
            for index, score in enumerate(generated.scores)
        ]
        confidence = torch.stack(log_probs).mean().exp().item() if log_probs else 0.0
        raw = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        reference = reference_classes(label_path)
        prediction = predicted_classes(raw)
        records.append({
            "sample_id": label_path.stem, "reference": reference,
            "prediction": prediction, "raw_output": raw,
            "exact_match": prediction == reference,
            "confidence": confidence, "latency_ms": latency_ms,
        })
        print(json.dumps({"sample": label_path.stem, "raw": raw}), flush=True)

    distribution = Counter(name for row in records for name in row["prediction"])
    payload = {
        "claim_scope": "predeclared_second_family_on_frozen_agdd_test",
        "model_id": MODEL_ID, "subset": args.subset,
        "split_manifest": str(args.split_manifest), "case_count": len(records),
        "exact_match_accuracy": sum(row["exact_match"] for row in records) / len(records),
        "parse_rate": sum(bool(row["prediction"]) for row in records) / len(records),
        "mean_latency_ms": sum(row["latency_ms"] for row in records) / len(records),
        "predicted_class_counts": dict(sorted(distribution.items())),
        "classification_metrics": classification_metrics(records),
        "calibration_metrics": calibration_metrics(records), "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
