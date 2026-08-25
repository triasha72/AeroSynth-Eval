#!/usr/bin/env python3
"""QLoRA-adapt Qwen2.5-VL-3B on AGDD's native held-out task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NAMES = {0: "contusion", 1: "scratches", 2: "crack", 3: "spot"}
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "Inspect this aircraft glass canopy image. Return every visible defect class as a "
    "comma-separated list in numeric order, chosen only from: contusion, scratches, crack, spot."
)


def classes_in(path: Path) -> str:
    ids = sorted({int(line.split()[0]) for line in path.read_text().splitlines() if line})
    return ", ".join(CLASS_NAMES[class_id] for class_id in ids)


def build_dataset(root: Path, split: str):  # type: ignore[no-untyped-def]
    from datasets import Dataset, Image

    records = []
    for label_path in sorted((root / "data" / "labels" / split).glob("*.txt")):
        image_path = root / "data" / "image" / split / f"{label_path.stem}.png"
        records.append(
            {
                "image": str(image_path),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": classes_in(label_path)}],
                    },
                ],
            }
        )
    return Dataset.from_list(records).cast_column("image", Image())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from trl import SFTConfig, SFTTrainer

    train_dataset = build_dataset(args.dataset, "train")
    eval_dataset = build_dataset(args.dataset, "val")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=str(args.output),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        max_length=None,
        fp16=True,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=17,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora,
        processing_class=processor,
    )
    result = trainer.train()
    trainer.save_model(str(args.output))
    metrics = {
        "claim_scope": "agdd_native_task_qlora_training_smoke",
        "model_id": MODEL_ID,
        "dataset": "AGDD",
        "source_commit": "4b5daa92929934f30b1155033c3ce67b7701960f",
        "train_examples": len(train_dataset),
        "held_out_examples": len(eval_dataset),
        "max_steps": args.max_steps,
        "lora_rank": 8,
        "lora_alpha": 16,
        "metrics": result.metrics,
    }
    (args.output / "training_evidence.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
