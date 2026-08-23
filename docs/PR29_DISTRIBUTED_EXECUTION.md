# PR #29 — Distributed/sharded execution layer

## Purpose

Add executor-neutral distributed sharding and deterministic reduction so evaluation can scale beyond one local sequential process.

## Methodology boundary

- Do not claim results that have not actually been run.
- Keep GenAI-Bench/RichHF/public data out of AeroSynth's internal protected-test split.
- Do not fabricate human annotations or adjudications.
- Record model, dataset, prompt and code revisions for every reported result.

## Gate

The deterministic planner and strict reducer are implemented and tested. A
free Kaggle shared-session backend is also available for a real 12-case
development batch. A distributed-performance claim still requires executing
every planned shard, preserving its manifests, and successfully reducing the
complete expected ID set.
