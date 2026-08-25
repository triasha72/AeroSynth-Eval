#!/usr/bin/env python3
"""Evaluate base or QLoRA-adapted Qwen on held-out AGDD validation images."""

from __future__ import annotations

import argparse
import json
import math
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


def classification_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute deterministic multilabel metrics without optional ML dependencies."""
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    per_class = {}
    totals = Counter()
    for name in CLASS_NAMES.values():
        true_positive = false_positive = false_negative = 0
        for row in records:
            reference = set(row["reference"])
            prediction = set(row["prediction"])
            true_positive += int(name in reference and name in prediction)
            false_positive += int(name not in reference and name in prediction)
            false_negative += int(name in reference and name not in prediction)
        precision = ratio(true_positive, true_positive + false_positive)
        recall = ratio(true_positive, true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }
        totals.update(tp=true_positive, fp=false_positive, fn=false_negative)
    macro_f1 = sum(metric["f1"] for metric in per_class.values()) / len(per_class)
    micro_precision = ratio(totals["tp"], totals["tp"] + totals["fp"])
    micro_recall = ratio(totals["tp"], totals["tp"] + totals["fn"])
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {"per_class": per_class, "macro_f1": macro_f1, "micro_f1": micro_f1}


def calibration_metrics(
    records: list[dict[str, object]], bins: int = 5
) -> dict[str, object]:
    """Summarize sequence-confidence calibration against exact-set correctness."""
    points = [(float(row["confidence"]), bool(row["exact_match"])) for row in records]
    epsilon = 1e-7
    brier = sum((confidence - float(correct)) ** 2 for confidence, correct in points) / len(points)
    negative_log_likelihood = -sum(
        math.log(max(epsilon, min(1 - epsilon, confidence)))
        if correct
        else math.log(max(epsilon, min(1 - epsilon, 1 - confidence)))
        for confidence, correct in points
    ) / len(points)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for point in points:
        buckets[min(bins - 1, int(point[0] * bins))].append(point)
    expected_calibration_error = sum(
        len(bucket)
        / len(points)
        * abs(
            sum(confidence for confidence, _ in bucket) / len(bucket)
            - sum(correct for _, correct in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )
    return {
        "confidence_definition": "geometric mean probability of generated tokens",
        "mean_confidence": sum(confidence for confidence, _ in points) / len(points),
        "brier_score": brier,
        "negative_log_likelihood": negative_log_likelihood,
        "expected_calibration_error": expected_calibration_error,
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--subset", choices=("selection", "test"))
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
    allowed_ids = None
    if bool(args.split_manifest) != bool(args.subset):
        raise SystemExit("--split-manifest and --subset must be provided together")
    if args.split_manifest:
        split_payload = json.loads(args.split_manifest.read_text())
        allowed_ids = set(split_payload[f"{args.subset}_ids"])
    labels = args.dataset / "data" / "labels" / "val"
    for label_path in sorted(labels.glob("*.txt")):
        if allowed_ids is not None and label_path.stem not in allowed_ids:
            continue
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
            generated = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        sequences = generated.sequences
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, sequences, strict=True)
        ]
        generated_tokens = sequences[:, inputs.input_ids.shape[1] :]
        token_log_probabilities = []
        for index, scores in enumerate(generated.scores):
            token = generated_tokens[:, index]
            token_log_probabilities.append(
                scores.log_softmax(dim=-1).gather(1, token[:, None]).squeeze(1)
            )
        confidence = (
            torch.stack(token_log_probabilities).mean().exp().item()
            if token_log_probabilities
            else 0.0
        )
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
                "confidence": confidence,
                "latency_ms": latency_ms,
            }
        )
        print(json.dumps({"sample": label_path.stem, "raw": raw}), flush=True)

    predicted_distribution = Counter(name for row in records for name in row["prediction"])
    payload = {
        "claim_scope": "agdd_native_held_out_validation",
        "model_id": MODEL_ID,
        "adapter": str(args.adapter) if args.adapter else None,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "subset": args.subset,
        "case_count": len(records),
        "exact_match_accuracy": sum(row["exact_match"] for row in records) / len(records),
        "parse_rate": sum(bool(row["prediction"]) for row in records) / len(records),
        "mean_latency_ms": sum(row["latency_ms"] for row in records) / len(records),
        "predicted_class_counts": dict(sorted(predicted_distribution.items())),
        "classification_metrics": classification_metrics(records),
        "calibration_metrics": calibration_metrics(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
