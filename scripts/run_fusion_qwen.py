#!/usr/bin/env python3
"""Evaluate vision/transcript fusion and missing/noisy transcript robustness."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_temporal_qwen import CONDITIONS, MODEL_ID, discover_cases, materialize_frames, percentile

MODES = (
    "vision_only",
    "transcript_only",
    "fusion",
    "missing_transcript",
    "noisy_transcript",
)
TRANSCRIPTS = {
    "clean": "Operator note: surface appears uniform with no visible anomaly.",
    "corrosion": "Operator note: localized oxidation and pitting are visible.",
    "crack": "Operator note: a thin linear discontinuity crosses the surface.",
    "coating": "Operator note: the finish layer appears peeled and discolored.",
}
PROMPT = "Return exactly one condition from: clean, corrosion, crack, coating."


def noisy_transcript(text: str) -> str:
    words = text.split()
    return " ".join("[inaudible]" if index % 3 == 1 else word for index, word in enumerate(words))


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["mode"])].append(record)
    summaries = {}
    for mode, rows in sorted(grouped.items()):
        latencies = [float(row["latency_ms"]) for row in rows]
        failures = Counter(str(row["reference"]) for row in rows if not bool(row["correct"]))
        summaries[mode] = {
            "case_count": len(rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": percentile(latencies, 0.95),
            "failure_counts_by_condition": dict(sorted(failures.items())),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("data/assets/v0_1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=32 * 28 * 28, max_pixels=96 * 28 * 28
    )
    cases = discover_cases(args.assets)
    records = []
    with tempfile.TemporaryDirectory(prefix="aerosynth-fusion-") as temporary:
        temp_root = Path(temporary)
        for case_id, (reference, source_paths) in cases.items():
            frames = materialize_frames(source_paths, 4, temp_root / case_id)
            for mode in MODES:
                content = []
                if mode != "transcript_only":
                    content.extend(
                        {"type": "image", "image": f"file://{path.resolve()}"}
                        for path in frames
                    )
                transcript = ""
                if mode in {"transcript_only", "fusion"}:
                    transcript = TRANSCRIPTS[reference]
                elif mode == "noisy_transcript":
                    transcript = noisy_transcript(TRANSCRIPTS[reference])
                elif mode == "missing_transcript":
                    transcript = "Operator transcript unavailable."
                text_prompt = f"{transcript}\n{PROMPT}" if transcript else PROMPT
                content.append({"type": "text", "text": text_prompt})
                messages = [{"role": "user", "content": content}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                images, videos = process_vision_info(messages)
                inputs = processor(
                    text=[rendered],
                    images=images or None,
                    videos=videos or None,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                latency_ms = (time.perf_counter() - started) * 1000
                trimmed = generated[:, inputs.input_ids.shape[1] :]
                raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                prediction = next(
                    (condition for condition in CONDITIONS if condition in raw.lower()), None
                )
                record = {
                    "case_id": case_id,
                    "mode": mode,
                    "reference": reference,
                    "prediction": prediction,
                    "raw_output": raw,
                    "correct": prediction == reference,
                    "latency_ms": latency_ms,
                }
                records.append(record)
                print(json.dumps(record), flush=True)

    payload = {
        "claim_scope": "synthetic_operator_transcript_fusion_robustness",
        "model_id": MODEL_ID,
        "case_count": len(cases),
        "modes": list(MODES),
        "transcript_provenance": (
            "Deterministic condition-grounded operator-note templates; no waveform or ASR stage."
        ),
        "summaries": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
