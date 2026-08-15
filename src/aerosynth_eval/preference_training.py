"""Prepare leakage-controlled preference-supervised data and MLX-VLM LoRA run manifests."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aerosynth_eval.preference_benchmark import (
    BenchmarkPartition,
    PreferenceExample,
    PreferenceLabel,
)


class AdaptationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_model_id: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    seed: int = 0


def _assistant_label(label: PreferenceLabel) -> str:
    return json.dumps({"preference": label.value}, sort_keys=True)


def training_messages(example: PreferenceExample) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example.left_image},
                {"type": "image", "image": example.right_image},
                {
                    "type": "text",
                    "text": "Choose the human-preferred candidate for this request: "
                    + example.prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": _assistant_label(example.human_preference)}],
        },
    ]


def prepare_mlx_vlm_dataset(examples: list[PreferenceExample], output_dir: Path) -> Path:
    """Save a local Hugging Face DatasetDict compatible with MLX-VLM training."""
    try:
        from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        raise ValueError(
            'Install training extras: python -m pip install -e ".[benchmark,train]"'
        ) from error

    partition_rows: dict[str, dict[str, list[object]]] = {
        BenchmarkPartition.TRAIN.value: {
            "images": [],
            "messages": [],
        },
    }

    for example in examples:
        if example.partition is not BenchmarkPartition.TRAIN:
            continue

        train_rows = partition_rows[BenchmarkPartition.TRAIN.value]
        train_rows["images"].append(
            [
                example.left_image,
                example.right_image,
            ]
        )
        train_rows["messages"].append(training_messages(example))
    dataset = DatasetDict(
        {key: Dataset.from_dict(rows) for key, rows in partition_rows.items() if rows["messages"]}
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite training dataset '{output_dir}'.")
    dataset.save_to_disk(str(output_dir))
    return output_dir


def write_adaptation_config(config: AdaptationConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def training_command(config: AdaptationConfig, dataset_path: Path, output_path: Path) -> list[str]:
    """Return an auditable command vector; execution remains explicit and user-controlled."""
    return [
        "python",
        "-m",
        "mlx_vlm.lora",
        "--model-path",
        config.base_model_id,
        "--dataset",
        str(dataset_path),
        "--split",
        BenchmarkPartition.TRAIN.value,
        "--batch-size",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--train-on-completions",
        "--grad-checkpoint",
        "--output-path",
        str(output_path),
    ]
