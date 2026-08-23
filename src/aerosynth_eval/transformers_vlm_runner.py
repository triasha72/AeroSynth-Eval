"""Linux/CUDA Transformers session for free Kaggle development evaluation."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

from aerosynth_eval.autograder import AutograderRequest
from aerosynth_eval.mlx_vlm_runner import (
    MlxVlmFailureKind,
    MlxVlmRequestBoundInference,
    MlxVlmRunConfig,
    MlxVlmRunnerError,
)

DEFAULT_TRANSFORMERS_VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def create_transformers_vlm_session(
    config: MlxVlmRunConfig,
) -> MlxVlmRequestBoundInference:
    """Load one 4-bit CUDA model and return reusable request-bound inference."""

    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
        qwen_utils = import_module("qwen_vl_utils")
    except ModuleNotFoundError as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            'Install the Kaggle backend with: pip install -e ".[kaggle]"',
        ) from error

    if not torch.cuda.is_available():
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            "The Transformers VLM backend requires a CUDA GPU.",
        )

    try:
        quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
        )
        processor = transformers.AutoProcessor.from_pretrained(config.model_id)
    except Exception as error:
        raise MlxVlmRunnerError(
            MlxVlmFailureKind.MODEL_LOAD_FAILURE,
            f"Transformers VLM model/session initialization failed: {error}",
        ) from error

    process_vision_info = cast(Any, qwen_utils.process_vision_info)

    def inference(
        runtime_config: MlxVlmRunConfig,
        prompt: str,
        image_path: Path,
        _: AutograderRequest,
    ) -> str:
        if runtime_config != config:
            raise ValueError("Shared Transformers session configuration changed during a batch.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[rendered],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=runtime_config.max_tokens,
                do_sample=runtime_config.temperature > 0,
                temperature=max(runtime_config.temperature, 1e-5),
            )
            trimmed = [
                output[len(source) :]
                for source, output in zip(inputs.input_ids, generated, strict=True)
            ]
            text = cast(
                str,
                processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip(),
            )
        except Exception as error:
            raise MlxVlmRunnerError(
                MlxVlmFailureKind.INFERENCE_FAILURE,
                f"Transformers VLM inference failed: {error}",
            ) from error
        if not text:
            raise MlxVlmRunnerError(
                MlxVlmFailureKind.EMPTY_RESPONSE,
                "Transformers VLM returned no non-empty text output.",
            )
        return text

    return inference
