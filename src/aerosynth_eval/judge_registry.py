"""Pluggable judge registry for cross-model evaluation on a common benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class JudgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    judge_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    family: str
    model_id: str
    model_revision: str = "main"
    enabled: bool = True
    notes: str = ""


class JudgeBackend(Protocol):
    spec: JudgeSpec

    def judge(self, prompt: str, left_image: Path, right_image: Path) -> str: ...


def load_judge_specs(path: Path) -> list[JudgeSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = [JudgeSpec.model_validate(item) for item in payload]
    enabled = [spec for spec in specs if spec.enabled]
    if len({spec.family for spec in enabled}) < 2:
        raise ValueError(
            "Enable at least two distinct model families for a multi-judge comparison."
        )
    return enabled


def comparison_key(spec: JudgeSpec) -> str:
    return f"{spec.judge_id}:{spec.model_id}@{spec.model_revision}"
