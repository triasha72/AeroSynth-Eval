#!/usr/bin/env python3
"""Completion-only SmolVLM2 QLoRA on AeBAD-V video1 windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_aebad_video_smolvlm2 import MODEL_ID, PROMPT, sampled_frames


def build_records(
    dataset: Path, cases: list[dict[str, object]], repetitions: int
) -> list[dict[str, object]]:
    records = []
    for case in cases:
        paths = sampled_frames(case["frames"], 4)
        prompt = [{"role": "user", "content": [
            *[{"type": "image"} for _ in paths],
            {"type": "text", "text": PROMPT},
        ]}]
        completion = [{
            "role": "assistant",
            "content": [{"type": "text", "text": str(case["label"]).upper()}],
        }]
        record = {
            "images": [str((dataset / path).resolve()) for path in paths],
            "prompt": prompt,
            "completion": completion,
        }
        records.extend([record] * repetitions)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    import torch
    from datasets import Dataset, Image, List
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, SmolVLMForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    cases = json.loads(args.manifest.read_text())["selection_cases"]
    train_cases = [case for case in cases if not str(case["case_id"]).endswith("-4")]
    eval_cases = [case for case in cases if str(case["case_id"]).endswith("-4")]
    train = Dataset.from_list(build_records(args.dataset, train_cases, 8)).cast_column(
        "images", List(Image())
    )
    evaluate = Dataset.from_list(build_records(args.dataset, eval_cases, 1)).cast_column(
        "images", List(Image())
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16, quantization_config=quantization
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, do_image_splitting=False, size={"longest_edge": 384}
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    config = SFTConfig(
        output_dir=str(args.output), max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, learning_rate=2e-4,
        max_length=None, fp16=False, logging_steps=1, eval_strategy="steps", eval_steps=10,
        save_strategy="steps", save_steps=10, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, completion_only_loss=True,
        report_to="none", seed=17, dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=train, eval_dataset=evaluate,
        processing_class=processor,
    )
    result = trainer.train()
    trainer.save_model(str(args.output))
    evidence = {
        "claim_scope": "aebad_video1_four_frame_completion_only_qlora",
        "model_id": MODEL_ID, "train_cases": len(train_cases), "eval_cases": len(eval_cases),
        "expanded_train_examples": len(train), "max_steps": args.max_steps,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric, "metrics": result.metrics,
    }
    (args.output / "training_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
