"""Scalable evaluation primitives: deterministic shards, cache keys, checkpoints, and manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    dataset_revision: str
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: str
    model_revision: str
    prompt_version: str
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    num_shards: int = Field(ge=1)


class EvaluationExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str = Field(pattern=r"^eval-[a-f0-9]{16}$")
    created_at: datetime
    git_commit: str
    config: EvaluationRunConfig


def experiment_id(config: EvaluationRunConfig, git_commit: str) -> str:
    payload = json.dumps(
        {"config": config.model_dump(mode="json"), "git_commit": git_commit}, sort_keys=True
    )
    return f"eval-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def build_experiment_manifest(
    config: EvaluationRunConfig, git_commit: str
) -> EvaluationExperimentManifest:
    return EvaluationExperimentManifest(
        experiment_id=experiment_id(config, git_commit),
        created_at=datetime.now(UTC).replace(microsecond=0),
        git_commit=git_commit,
        config=config,
    )


def shard_index(example_id: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    return int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) % num_shards


def cache_key(
    *, example_id: str, model_id: str, model_revision: str, prompt_version: str, config_payload: str
) -> str:
    raw = "|".join((example_id, model_id, model_revision, prompt_version, config_payload))
    return hashlib.sha256(raw.encode()).hexdigest()


class JsonlCheckpointStore:
    """Append-only success checkpoint. Failures are not silently cached as successes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def completed_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        completed: set[str] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    payload = json.loads(line)
                    if payload.get("status") == "success":
                        completed.add(str(payload["example_id"]))
        return completed

    def append_success(self, example_id: str, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"example_id": example_id, "status": "success", "payload": payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
