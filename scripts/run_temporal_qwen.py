#!/usr/bin/env python3
"""Run 1/4/8/16-frame Qwen ablations on derived AeroSynth inspection sequences."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
FRAME_COUNTS = (1, 4, 8, 16)
CONDITIONS = ("clean", "corrosion", "crack", "coating")
PROMPT = (
    "These ordered frames are from one short aircraft surface inspection. Use temporal context "
    "across all frames and return exactly one condition from: clean, corrosion, crack, coating."
)


def discover_cases(asset_root: Path) -> dict[str, tuple[str, list[Path]]]:
    cases: dict[str, tuple[str, list[Path]]] = {}
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(asset_root.glob("*.png")):
        region, condition, *_ = path.stem.split("-")
        grouped[(region, condition)].append(path)
    for (region, condition), paths in sorted(grouped.items()):
        case_id = f"{region}-{condition}"
        cases[case_id] = (condition, paths)
    return cases


def materialize_frames(source_paths: list[Path], count: int, output_dir: Path) -> list[Path]:
    """Create ordered camera-motion variants without changing semantic labels."""
    if not source_paths:
        raise ValueError("At least one source frame is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(count):
        source = source_paths[index % len(source_paths)]
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            inset = index % 4
            cropped = image.crop((inset, inset, width - inset, height - inset)).resize(
                (width, height), Image.Resampling.BICUBIC
            )
            brightness = 0.94 + 0.02 * (index % 7)
            frame = ImageEnhance.Brightness(cropped).enhance(brightness)
            destination = output_dir / f"frame-{index:02d}.jpg"
            frame.save(destination, quality=90)
            frames.append(destination)
    return frames


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    by_count: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_count[int(record["frame_count"])].append(record)
    summaries = {}
    for count, rows in sorted(by_count.items()):
        latencies = [float(row["latency_ms"]) for row in rows]
        failure_counts = Counter(
            str(row["reference"]) for row in rows if not bool(row["correct"])
        )
        summaries[str(count)] = {
            "case_count": len(rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": percentile(latencies, 0.95),
            "failure_counts_by_condition": dict(sorted(failure_counts.items())),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("data/assets/v0_1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    cases = list(discover_cases(args.assets).items())
    if args.limit:
        cases = cases[: args.limit]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=32 * 28 * 28, max_pixels=96 * 28 * 28
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="aerosynth-temporal-") as temporary:
        temp_root = Path(temporary)
        for case_id, (reference, source_paths) in cases:
            for count in FRAME_COUNTS:
                frame_paths = materialize_frames(
                    source_paths, count, temp_root / case_id / str(count)
                )
                content = [
                    {"type": "image", "image": f"file://{path.resolve()}"}
                    for path in frame_paths
                ]
                content.append({"type": "text", "text": PROMPT})
                messages = [{"role": "user", "content": content}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                images, videos = process_vision_info(messages)
                inputs = processor(
                    text=[rendered],
                    images=images,
                    videos=videos,
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
                    "frame_count": count,
                    "reference": reference,
                    "prediction": prediction,
                    "raw_output": raw,
                    "correct": prediction == reference,
                    "latency_ms": latency_ms,
                    "derived_sequence": True,
                }
                records.append(record)
                print(json.dumps(record), flush=True)

    payload = {
        "claim_scope": "derived_multiframe_aerosynth_temporal_ablation",
        "model_id": MODEL_ID,
        "frame_counts": list(FRAME_COUNTS),
        "case_count": len(cases),
        "sequence_provenance": (
            "Deterministic camera-motion and brightness variants derived from four independently "
            "rendered capture profiles per region-condition case; not real captured video."
        ),
        "summaries": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
