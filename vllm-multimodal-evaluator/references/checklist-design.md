# Checklist Design

Use this reference when adjusting the evaluator or explaining a run result.

## Design goal

The evaluator is a capability gate, not a benchmark.

Its job is to separate:

- media ingestion support
- basic multimodal semantic support
- function-calling support
- transport-specific issues

## Standard coverage

Standard capability runs default to all of the following:

1. `phase1_ingestion_standard`
2. `phase2_semantic_standard`
3. `phase1_ingestion_large_image_smoke`
4. `phase2_semantic_large_image_smoke`
5. `phase2_function_calling_standard`

Standard transport coverage:

- `local_path`
- `base64`
- `http`

## What each phase means

### Phase 1

Phase 1 checks whether the service can ingest the media correctly.

Typical cases:

- image format ingestion
- image resolution ingestion
- video format ingestion
- video resolution ingestion

### Phase 2

Phase 2 checks whether the model understands the already-ingested media.

Typical cases:

- single-image semantic understanding
- image resolution semantic understanding
- multi-image understanding
- interleaved text/image ordering
- video semantic understanding
- video order and detail understanding

### Large-image smoke

Large-image smoke is intentionally light:

- `4096x4096`
- `4096x6144`
- `4096x8192`

It exists to catch obvious regressions in high-resolution ingestion and simple semantic understanding.

### Function calling

Function calling is included by default so the evaluator can report a separate capability signal for tool-use related output behavior.

## Transport policy

The evaluator should report transport results using the user-facing mode names:

- `local_path`
- `base64`
- `http`

Internally, preserve the URL implementation detail when relevant:

- `file_url`
- `data_url`
- `http_url`

## PASS and FAIL

Use the phase split for attribution:

- `Phase 1 FAIL` first suggests a serving, path, permission, MIME, or static-file issue.
- `Phase 2 FAIL` after successful ingestion first suggests a model capability issue.
- `unknown` or explanation-heavy output first suggests output protocol or extraction instability.

## Reporting

Reports should always include:

- service config
- selected transport modes
- enabled optional suites
- included test items
- counts by suite
- counts by media scale
- counts by transport mode
- `failure_class`
- `root_cause_note`

## Practical rule

If a failure was fixed by repairing evaluator transport logic, static media serving, or dataset wiring, do not keep counting that historical failure as a model defect in later reports.
