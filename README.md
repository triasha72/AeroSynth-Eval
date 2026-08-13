# AeroSynth-Eval

AeroSynth-Eval is an independent research and engineering project exploring how
visual-language models can evaluate synthetic aerospace inspection imagery against
explicit, human-reviewable rubrics.

## Research question

Can an evaluator determine whether a generated aircraft-exterior inspection image
matches its requested context and condition, while identifying when the image should
be rejected or escalated for human review?

## Scope

The v0.1 rubric evaluates:

- aircraft-region and inspection-context fidelity;
- requested surface-condition fidelity;
- visual quality, including focus, lighting, framing, glare, and occlusion;
- suitability for a non-operational inspection-data workflow.

All examples in this project must use public, synthetic, or self-generated data only.

## Safety and limitations

This is an independent research prototype. It is not an airworthiness,
maintenance-release, defect-diagnosis, or certification system. Its outputs support
research dataset quality review and must not replace qualified human inspection.

## Current status

v0.1 establishes typed data contracts, a fixed rubric, CLI access, and regression
tests. It does not claim VLM evaluation accuracy, human agreement, calibration, or
fine-tuning results.

## Development

```bash
conda activate aerosynth-eval-py312
python -m pip install -e ".[dev]"

python -m ruff format .
python -m ruff check .
python -m mypy src/aerosynth_eval
python -m pytest

aerosynth-eval info
aerosynth-eval rubric
```
