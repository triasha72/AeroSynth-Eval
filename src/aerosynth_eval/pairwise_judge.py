"""Pairwise multimodal judge contracts and local MLX-VLM execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosynth_eval.preference_benchmark import PreferenceExample, PreferenceLabel


class JudgeOrientation(StrEnum):
    STANDARD = "standard"
    SWAPPED = "swapped"


class PairwiseJudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preference: PreferenceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=4_000)


class PairwiseJudgmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str
    orientation: JudgeOrientation
    model_id: str
    human_preference: PreferenceLabel
    predicted_preference: PreferenceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=4_000)
    raw_output: str = Field(min_length=1, max_length=20_000)


JudgeInference = Callable[[str, Path, Path], str]


def swap_label(label: PreferenceLabel) -> PreferenceLabel:
    if label is PreferenceLabel.A_PREFERRED:
        return PreferenceLabel.B_PREFERRED
    if label is PreferenceLabel.B_PREFERRED:
        return PreferenceLabel.A_PREFERRED
    return label


def render_pairwise_prompt(example: PreferenceExample, orientation: JudgeOrientation) -> str:
    del orientation
    return (
        "You are evaluating two AI-generated images against the user's request. "
        "Judge prompt adherence, visual quality, artifacts, and overall usefulness. "
        "Return only a JSON object matching the supplied schema.\n\n"
        f"REQUEST:\n{example.prompt}\n\n"
        "Candidate A is always the first presented image; "
        "candidate B is always the second presented image."
    )


def parse_response(raw_output: str) -> PairwiseJudgeResponse:
    try:
        payload: object = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError("Pairwise judge output must be exactly one JSON object.") from error
    try:
        return PairwiseJudgeResponse.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"Pairwise judge output violates schema: {error}") from error


def _extract_text(output: object) -> str:
    text: object

    if isinstance(output, str):
        text = output
    else:
        text = getattr(output, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("MLX-VLM returned an empty pairwise response.")
    return text


def create_mlx_pairwise_session(
    model_id: str, *, max_tokens: int = 500, temperature: float = 0.0
) -> JudgeInference:
    """Load a VLM once and return multi-image pairwise inference."""

    mlx_vlm = import_module("mlx_vlm")
    prompt_utils = import_module("mlx_vlm.prompt_utils")
    structured = import_module("mlx_vlm.structured")
    utils = import_module("mlx_vlm.utils")
    load = cast(Callable[[str], tuple[Any, Any]], mlx_vlm.load)
    generate = cast(Callable[..., object], mlx_vlm.generate)
    apply_chat_template = cast(Callable[..., str], prompt_utils.apply_chat_template)
    build_schema = cast(
        Callable[[Any, dict[str, Any]], Any], structured.build_json_schema_logits_processor
    )
    load_config = cast(Callable[[str], Any], utils.load_config)
    model, processor = load(model_id)
    config = load_config(model_id)
    tokenizer = getattr(processor, "tokenizer", processor)
    schema = PairwiseJudgeResponse.model_json_schema()

    def infer(prompt: str, left_image: Path, right_image: Path) -> str:
        logits_processor = build_schema(tokenizer, schema)
        formatted = apply_chat_template(processor, config, prompt, num_images=2)
        output = generate(
            model,
            processor,
            formatted,
            image=[str(left_image), str(right_image)],
            verbose=False,
            max_tokens=max_tokens,
            temperature=temperature,
            logits_processors=[logits_processor],
        )
        return _extract_text(output)

    return infer


def judge_example(
    example: PreferenceExample,
    *,
    model_id: str,
    inference: JudgeInference,
    orientation: JudgeOrientation = JudgeOrientation.STANDARD,
) -> PairwiseJudgmentRecord:
    if orientation is JudgeOrientation.STANDARD:
        left_image = Path(example.left_image)
        right_image = Path(example.right_image)
        human = example.human_preference
    else:
        left_image = Path(example.right_image)
        right_image = Path(example.left_image)
        human = swap_label(example.human_preference)
    raw = inference(render_pairwise_prompt(example, orientation), left_image, right_image)
    response = parse_response(raw)
    return PairwiseJudgmentRecord(
        example_id=example.example_id,
        orientation=orientation,
        model_id=model_id,
        human_preference=human,
        predicted_preference=response.preference,
        confidence=response.confidence,
        rationale=response.rationale,
        raw_output=raw,
    )
