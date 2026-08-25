#!/usr/bin/env python3
"""Replay AeroSynth frames through a bounded, sliding-window Qwen inference path."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

from run_temporal_qwen import CONDITIONS, MODEL_ID, discover_cases, materialize_frames, percentile

PROMPT = "Return exactly one condition from: clean, corrosion, crack, coating."


@dataclass
class BoundedDropOldestQueue[T]:
    capacity: int
    items: deque[T] = field(default_factory=deque)
    dropped: int = 0

    def put(self, item: T) -> None:
        if len(self.items) >= self.capacity:
            self.items.popleft()
            self.dropped += 1
        self.items.append(item)

    def get(self) -> T:
        return self.items.popleft()


def summarize_mode(records: list[dict[str, object]], wall_seconds: float) -> dict[str, object]:
    latencies = [float(record["latency_ms"]) for record in records]
    failures = Counter(
        str(record["reference"]) for record in records if not bool(record["correct"])
    )
    return {
        "processed_windows": len(records),
        "accuracy": sum(bool(record["correct"]) for record in records) / len(records),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "throughput_windows_per_second": len(records) / wall_seconds,
        "failure_counts_by_condition": dict(sorted(failures.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("data/assets/v0_1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-capacity", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=4)
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
    torch.cuda.reset_peak_memory_stats()

    def infer(window: deque[Path], reference: str, case_id: str, mode: str) -> dict[str, object]:
        content = [
            {"type": "image", "image": f"file://{path.resolve()}"} for path in window
        ]
        content.append({"type": "text", "text": PROMPT})
        messages = [{"role": "user", "content": content}]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        latency_ms = (time.perf_counter() - started) * 1000
        raw = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        prediction = next((name for name in CONDITIONS if name in raw.lower()), None)
        return {
            "mode": mode,
            "case_id": case_id,
            "reference": reference,
            "prediction": prediction,
            "correct": prediction == reference,
            "window_frames": len(window),
            "latency_ms": latency_ms,
        }

    all_records = []
    mode_summaries = {}
    with tempfile.TemporaryDirectory(prefix="aerosynth-stream-") as temporary:
        temp_root = Path(temporary)
        materialized = {
            case_id: (
                reference,
                materialize_frames(paths, 4, temp_root / case_id),
            )
            for case_id, (reference, paths) in discover_cases(args.assets).items()
        }
        for mode in ("normal", "overload", "injected_drop"):
            records = []
            total_dropped = 0
            recovered_cases = 0
            started_mode = time.perf_counter()
            for case_id, (reference, frames) in materialized.items():
                queue: BoundedDropOldestQueue[Path] = BoundedDropOldestQueue(args.queue_capacity)
                window: deque[Path] = deque(maxlen=args.window_size)
                processed_after_drop = False
                for index, frame in enumerate(frames):
                    if mode == "injected_drop" and index == 1:
                        total_dropped += 1
                        continue
                    queue.put(frame)
                    if mode == "normal":
                        window.append(queue.get())
                        record = infer(window, reference, case_id, mode)
                        records.append(record)
                        processed_after_drop = processed_after_drop or mode == "injected_drop"
                if mode != "normal":
                    had_drop = total_dropped > 0 or queue.dropped > 0
                    total_dropped += queue.dropped
                    while queue.items:
                        window.append(queue.get())
                        records.append(infer(window, reference, case_id, mode))
                        processed_after_drop = processed_after_drop or had_drop
                recovered_cases += int(processed_after_drop)
            wall_seconds = time.perf_counter() - started_mode
            summary = summarize_mode(records, wall_seconds)
            summary.update(
                {
                    "dropped_inputs": total_dropped,
                    "recovered_cases": recovered_cases,
                    "case_count": len(materialized),
                }
            )
            mode_summaries[mode] = summary
            all_records.extend(records)
            print(json.dumps({"mode": mode, **summary}), flush=True)

    payload = {
        "claim_scope": "recorded_frame_streaming_replay",
        "model_id": MODEL_ID,
        "queue_policy": "bounded drop-oldest",
        "queue_capacity": args.queue_capacity,
        "sliding_window_size": args.window_size,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "summaries": mode_summaries,
        "records": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
