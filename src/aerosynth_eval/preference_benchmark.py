"""External human-preference benchmark contracts and GenAI-Bench materialization.

This module keeps public human-preference benchmarks separate from AeroSynth's
synthetic development and protected-test splits. GenAI-Bench is treated as an
external benchmark and is never written into the internal asset registry.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

GENAI_BENCH_DATASET = "TIGER-Lab/GenAI-Bench"
GENAI_BENCH_REVISION = "main"
GENAI_BENCH_SPLIT = "test_v1"
GENAI_BENCH_LICENSE = "CC BY 4.0"


class PreferenceLabel(StrEnum):
    A_PREFERRED = "a_preferred"
    B_PREFERRED = "b_preferred"
    TIE_GOOD = "tie_good"
    TIE_BAD = "tie_bad"


class PreferenceTask(StrEnum):
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_edition"


class BenchmarkPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELDOUT = "heldout"


class PreferenceExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(pattern=r"^genai-[a-f0-9]{16}$")
    task: PreferenceTask
    partition: BenchmarkPartition
    prompt: str = Field(min_length=1, max_length=20_000)
    left_model: str = Field(min_length=1, max_length=200)
    right_model: str = Field(min_length=1, max_length=200)
    human_preference: PreferenceLabel
    left_image: str = Field(min_length=1, max_length=1_000)
    right_image: str = Field(min_length=1, max_length=1_000)
    source_dataset: str = GENAI_BENCH_DATASET
    source_revision: str = GENAI_BENCH_REVISION
    source_split: str = GENAI_BENCH_SPLIT
    source_license: str = GENAI_BENCH_LICENSE


class PreferenceManifestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int = Field(ge=0)
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    heldout_count: int = Field(ge=0)
    task_counts: dict[str, int]
    label_counts: dict[str, int]


def map_genai_vote(vote_type: str) -> PreferenceLabel:
    mapping = {
        "leftvote": PreferenceLabel.A_PREFERRED,
        "rightvote": PreferenceLabel.B_PREFERRED,
        "tievote": PreferenceLabel.TIE_GOOD,
        "bothbad_vote": PreferenceLabel.TIE_BAD,
    }
    try:
        return mapping[vote_type]
    except KeyError as error:
        raise ValueError(f"Unsupported GenAI-Bench vote_type '{vote_type}'.") from error


def partition_for_group(group_key: str) -> BenchmarkPartition:
    """Deterministically split related examples by prompt/group, preventing pair leakage."""

    bucket = int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return BenchmarkPartition.TRAIN
    if bucket < 85:
        return BenchmarkPartition.VALIDATION
    return BenchmarkPartition.HELDOUT


def _example_id(task: PreferenceTask, source_index: int, prompt: str) -> str:
    digest = hashlib.sha256(f"{task}:{source_index}:{prompt}".encode()).hexdigest()[:16]
    return f"genai-{digest}"


def _prompt_for_record(task: PreferenceTask, record: dict[str, Any]) -> str:
    if task is PreferenceTask.IMAGE_GENERATION:
        return str(record["prompt"]).strip()
    source = str(record["source_prompt"]).strip()
    target = str(record["target_prompt"]).strip()
    instruction = str(record["instruct_prompt"]).strip()
    return f"Source: {source}\nTarget: {target}\nEdit instruction: {instruction}"


def normalize_record(
    record: dict[str, Any],
    *,
    task: PreferenceTask,
    source_index: int,
    left_image_path: Path,
    right_image_path: Path,
) -> PreferenceExample:
    prompt = _prompt_for_record(task, record)
    group_key = prompt.casefold().strip()
    return PreferenceExample(
        example_id=_example_id(task, source_index, prompt),
        task=task,
        partition=partition_for_group(group_key),
        prompt=prompt,
        left_model=str(record["left_model"]),
        right_model=str(record["right_model"]),
        human_preference=map_genai_vote(str(record["vote_type"])),
        left_image=str(left_image_path),
        right_image=str(right_image_path),
    )


def write_manifest(records: list[PreferenceExample], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing manifest '{output_path}'.")
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
    return output_path


def load_manifest(path: Path) -> list[PreferenceExample]:
    records: list[PreferenceExample] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(PreferenceExample.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid preference record: {error}"
                ) from error
    if not records:
        raise ValueError(f"Preference manifest '{path}' contains no records.")
    return records


def summarize_manifest(records: list[PreferenceExample]) -> PreferenceManifestSummary:
    partitions = {partition.value: 0 for partition in BenchmarkPartition}
    tasks: dict[str, int] = {}
    labels: dict[str, int] = {}
    for record in records:
        partitions[record.partition.value] += 1
        tasks[record.task.value] = tasks.get(record.task.value, 0) + 1
        labels[record.human_preference.value] = labels.get(record.human_preference.value, 0) + 1
    return PreferenceManifestSummary(
        record_count=len(records),
        train_count=partitions[BenchmarkPartition.TRAIN.value],
        validation_count=partitions[BenchmarkPartition.VALIDATION.value],
        heldout_count=partitions[BenchmarkPartition.HELDOUT.value],
        task_counts=tasks,
        label_counts=labels,
    )


def materialize_genai_bench(
    output_root: Path,
    *,
    tasks: tuple[PreferenceTask, ...] = (
        PreferenceTask.IMAGE_GENERATION,
        PreferenceTask.IMAGE_EDITING,
    ),
    split: str = GENAI_BENCH_SPLIT,
    limit_per_task: int | None = None,
) -> tuple[Path, PreferenceManifestSummary]:
    """Download public data through Hugging Face and materialize images + normalized JSONL."""

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        raise ValueError(
            'Install benchmark extras: python -m pip install -e ".[benchmark]"'
        ) from error

    manifest_records: list[PreferenceExample] = []
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        dataset = load_dataset(GENAI_BENCH_DATASET, task.value, split=split)
        task_records = (
            dataset
            if limit_per_task is None
            else dataset.select(range(min(limit_per_task, len(dataset))))
        )
        for source_index, row in enumerate(task_records):
            record = dict(row)
            prompt = _prompt_for_record(task, record)
            example_id = _example_id(task, source_index, prompt)
            example_dir = image_root / task.value / example_id
            example_dir.mkdir(parents=True, exist_ok=True)
            left_path = example_dir / "left.png"
            right_path = example_dir / "right.png"
            left_key = (
                "left_image" if task is PreferenceTask.IMAGE_GENERATION else "left_output_image"
            )
            right_key = (
                "right_image" if task is PreferenceTask.IMAGE_GENERATION else "right_output_image"
            )
            record[left_key].convert("RGB").save(left_path)
            record[right_key].convert("RGB").save(right_path)
            manifest_records.append(
                normalize_record(
                    record,
                    task=task,
                    source_index=source_index,
                    left_image_path=left_path,
                    right_image_path=right_path,
                )
            )

    manifest_path = output_root / "genai_bench.preference.jsonl"
    write_manifest(manifest_records, manifest_path)
    return manifest_path, summarize_manifest(manifest_records)
