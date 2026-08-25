#!/usr/bin/env python3
"""Run native multi-frame SmolVLM2 ablations on frozen AeBAD-V windows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval_agdd_qwen_adapter import calibration_metrics

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
PROMPT = (
    "Inspect this ordered sequence of aero-engine blade frames. Is the blade sequence anomalous? "
    "Answer with exactly one word: anomaly or good."
)


def sampled_frames(frames: list[str], count: int) -> list[str]:
    if count == 1:
        return [frames[len(frames) // 2]]
    return [frames[round(index * (len(frames) - 1) / (count - 1))] for index in range(count)]


def parse_prediction(raw: str) -> str | None:
    lowered = raw.lower().strip()
    if "anomal" in lowered or "defect" in lowered:
        return "anomaly"
    if "good" in lowered or "normal" in lowered:
        return "good"
    return None


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    labels = ("good", "anomaly")
    recall = {}
    for label in labels:
        rows = [row for row in records if row["reference"] == label]
        recall[label] = sum(row["prediction"] == label for row in rows) / len(rows)
    return {
        "case_count": len(records),
        "accuracy": sum(row["exact_match"] for row in records) / len(records),
        "balanced_accuracy": sum(recall.values()) / len(recall),
        "per_class_recall": recall,
        "parse_rate": sum(row["prediction"] is not None for row in records) / len(records),
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in records) / len(records),
        "calibration_metrics": calibration_metrics(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("selection", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    import torch
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration

    manifest = json.loads(args.manifest.read_text())
    cases = manifest[f"{args.subset}_cases"]
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    ).eval()
    records = []
    for count in manifest["frame_count_ablation"]:
        for case in cases:
            paths = sampled_frames(case["frames"], count)
            content = [
                {"type": "image", "url": str((args.dataset / path).resolve())}
                for path in paths
            ]
            content.append({"type": "text", "text": PROMPT})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    return_dict_in_generate=True, output_scores=True,
                )
            latency = (time.perf_counter() - started) * 1000
            prompt_length = inputs["input_ids"].shape[-1]
            tokens = generated.sequences[:, prompt_length:]
            log_probs = [
                score.log_softmax(dim=-1).gather(1, tokens[:, index, None]).squeeze(1)
                for index, score in enumerate(generated.scores)
            ]
            confidence = torch.stack(log_probs).mean().exp().item() if log_probs else 0.0
            raw = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            prediction = parse_prediction(raw)
            records.append({
                "case_id": case["case_id"], "video": case["video"],
                "frame_count": count, "reference": case["label"],
                "prediction": prediction, "raw_output": raw,
                "exact_match": prediction == case["label"],
                "confidence": confidence, "latency_ms": latency,
            })
            print(json.dumps({"case": case["case_id"], "frames": count, "raw": raw}), flush=True)

    by_count = {
        str(count): summarize([row for row in records if row["frame_count"] == count])
        for count in manifest["frame_count_ablation"]
    }
    by_video = {
        video: summarize([row for row in records if row["video"] == video])
        for video in sorted({str(row["video"]) for row in records})
    }
    payload = {
        "claim_scope": "native_aebad_video_temporal_ablation",
        "model_id": MODEL_ID, "subset": args.subset,
        "manifest": str(args.manifest), "prompt": PROMPT,
        "by_frame_count": by_count, "by_video": by_video, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
