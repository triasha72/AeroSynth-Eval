# Evaluator output reliability

## Purpose

AeroSynth-Eval v0.12 strengthens the structured-output contract used by the
development-only VLM evaluator.

This milestone was motivated by an observed v0.11 development batch in which:

- 12 cases were attempted;
- 8 responses passed strict validation;
- 3 responses failed request binding because the generated `asset_id` dropped
  the required `asset-` prefix; and
- 1 response failed strict JSON parsing.

The v0.12 goal is not to hide those failures. It moves known request identity
constraints into generation while retaining strict post-generation validation
and better diagnostics for any response that is still rejected.

## Request-bound generation

The generic `AutograderResponse` schema still defines the public response
contract.

For each development request, AeroSynth-Eval now creates a request-bound copy of
that schema and pins:

- `asset_id` to the exact request asset;
- `scenario_id` to the exact request scenario;
- `rubric_version` to the exact rubric version; and
- `result_source` to `vlm_output`.

These fields are represented as single-value JSON-schema enums so the structured
decoder can constrain them during generation.

Evaluation-dependent fields remain model outputs:

- `decision`;
- `confidence`;
- four rubric scores;
- four rationales; and
- `summary`.

## Defense in depth

Request-bound generation does not replace validation.

The accepted path remains:

```text
development request
        |
        v
request-bound JSON schema
        |
        v
structured generation
        |
        v
strict JSON parsing
        |
        v
Pydantic response validation
        |
        v
strict request binding
        |
        v
accepted run record
```

The post-generation request-binding validator remains active so a decoder or
runtime regression cannot silently weaken identity guarantees.

## Rejected-output provenance

A rejected response is not repaired.

For JSON, schema, or request-binding failures, AeroSynth-Eval stores a bounded
diagnostic record containing:

- up to the first 4,000 characters of the rejected output;
- SHA-256 of the complete rejected output;
- complete character count; and
- whether the stored preview was truncated.

This provides enough provenance for debugging while avoiding unbounded rejected
text inside batch records.

The framework does not:

- strip Markdown fences and accept the result;
- extract a JSON-looking substring from surrounding prose;
- add a missing `asset-` prefix;
- replace a wrong scenario identifier; or
- otherwise rewrite rejected output into an accepted response.

## Development-only scope

This milestone continues to use only the frozen development queue.

The protected test split remains untouched.

The resulting execution-success rate remains an operational structured-response
measurement only. It is not evaluator accuracy, human agreement, calibration,
or inspection performance.

## Regression comparison

After v0.12 passes unit and integration validation, run one new real development
batch with the same model and runtime settings as the v0.11 baseline:

```bash
aerosynth-eval run-vlm-batch \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --max-tokens 800 \
  --temperature 0.0 \
  --max-retries 1
```

Compare the resulting failure taxonomy against the v0.11 baseline:

```text
request_binding_failure: 3
json_parse_failure:       1
```

The purpose is to determine whether request-bound generation removes identity
drift while still surfacing any genuinely malformed outputs.
