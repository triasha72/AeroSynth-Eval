from pathlib import Path

from aerosynth_eval.preference_training import AdaptationConfig, training_command


def test_training_command_is_explicit() -> None:
    config = AdaptationConfig(
        learning_rate=2e-5, batch_size=1, epochs=1, lora_rank=8, lora_alpha=16
    )
    command = training_command(config, Path("dataset"), Path("adapter.safetensors"))
    assert command[:3] == ["python", "-m", "mlx_vlm.lora"]
    assert "--train-on-completions" in command


def test_training_command_uses_train_partition_only() -> None:
    config = AdaptationConfig(
        learning_rate=2e-5,
        batch_size=1,
        epochs=1,
        lora_rank=8,
        lora_alpha=16,
    )

    command = training_command(
        config,
        Path("dataset"),
        Path("adapter.safetensors"),
    )

    split_index = command.index("--split")

    assert command[split_index + 1] == "train"
