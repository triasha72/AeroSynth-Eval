from pathlib import Path

from aerosynth_eval.transformers_vlm_runner import _transformers_messages


def test_transformers_messages_use_a_strict_system_instruction() -> None:
    image_path = Path("data/example.png")

    messages = _transformers_messages("request-bound prompt", image_path)

    assert messages[0]["role"] == "system"
    assert "complete object" in messages[0]["content"]
    assert "shortened score map" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": "request-bound prompt"},
        ],
    }
