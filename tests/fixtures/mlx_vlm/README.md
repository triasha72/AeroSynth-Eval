# MLX-VLM test boundary

This directory intentionally contains no recorded model output. The v0.9 unit
tests use an in-memory, clearly named test double to exercise the local-run
provenance and strict-response-validation code without downloading a model or
presenting synthetic text as VLM evidence.

No file in this directory is a VLM result, gold label, benchmark result,
human-rating record, or evaluator-quality claim.
