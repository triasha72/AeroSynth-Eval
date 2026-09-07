# System architecture

```mermaid
flowchart LR
    A[Public real images] --> B[Dataset audit and fixed splits]
    C[Generated inspection images] --> D[Scenario and asset registry]
    D --> E[VLM and human rating adapters]
    E --> F[Agreement and calibration]
    F --> G[Evaluator selection]
    G --> H[Synthetic subset selection]
    A --> I[Real-only baseline]
    H --> J[Real plus synthetic training]
    I --> K[Untouched real-image test]
    J --> K
    K --> L[Utility and release gates]
```

The real-image test split is kept separate from generated images and evaluator
selection. Synthetic data earns a positive result only when it improves an
untouched real benchmark without weakening the protected defect metric.

The current implementation uses local MLX, batched Transformers, and structured
JSON adapters behind the same evaluation contracts. Dataset licenses, source
checksums, experiment seeds, and non-claims travel with each result artifact.

As the project grows, the first change should be replacing the small AGDD
holdout with a larger, separately sourced aircraft dataset. Distributed judging
and model adaptation come after that evidence gap is closed.
