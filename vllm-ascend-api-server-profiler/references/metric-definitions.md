# Metric Definitions

## Timing terms

- `HTTP total wall time`
  End-to-end wall time for the profiled HTTP request on the API server side.
- `API server CPU time`
  CPU time consumed by the API server process during the profiled request window.
- `API server preprocess chain`
  The custom request span from `HfRenderer.render_messages_async` start to the final `SingleWriterShmObjectStorage.copy_to_buffer` end.
- `stage_spans`
  Boundary-defined sequential wall spans used as the primary API server breakdown.
- `concurrency_windows`
  First-start to last-end windows for repeated or overlapping functions that should not be read as simple sums of durations.

## Function timing labels

- `wall-critical-path`
  The elapsed wall span on the request timeline that blocks forward progress for the ordered step being discussed.
- `aggregate-total`
  The sum of all invocations for the same function. This may exceed wall time when many calls overlap.
- `inclusive-total`
  The elapsed time of one function invocation including its child calls. Do not add this directly to child totals.

## Stage-first interpretation

Treat `stage_spans` as the main API server wall-time view. A practical order is:

1. `api_server_total.preprocess_wall`
2. `api_stage.render_messages_wall`
3. `api_stage.tokenize_wall`
4. `api_stage.mm_hash_wall`
5. `api_stage.hf_processor_wall`
6. `api_stage.shm_copy_wall`

Use wrapper functions such as `InputProcessingContext.call_hf_processor` only to anchor a stage boundary. Use leaf functions such as `_preprocess`, `get_mm_hashes`, and `copy_to_buffer` to name the hotspot.

## Interpretation rules

### 1. Do not sum nested wrappers

Examples:

- `Qwen3VLMultiModalProcessor._call_hf_processor`
- `InputProcessingContext.call_hf_processor`
- `Qwen3VLProcessor.__call__`
- `Qwen2VLImageProcessorFast.preprocess`

These sit on the same stack and largely overlap. Use them to locate a stage, then drill down to the leaf hotspot.

### 2. Do not treat concurrent aggregate as wall time

Example:

- `MediaConnector.load_from_url_async` may show large `aggregate-total` across `40` calls while occupying a much smaller `wall-critical-path` because the loads overlap.
- `SingleWriterShmObjectStorage.copy_to_buffer` may show a `wall-critical-path` larger than `aggregate-total` because many short copies are spread across a wider first-start to last-end window with gaps between calls.

In other words, `aggregate-total` and `wall-critical-path` answer different questions and either one may be larger depending on overlap and gaps.

### 3. Explain stage gaps explicitly

The sum of the individual stage walls can be smaller than `api_server_total.preprocess_wall`.

Use:

- `sum(stage wall)` to describe the named stage spans
- `api_server_total.preprocess_wall` to describe the whole request-scoped preprocess window
- `gap = total preprocess wall - sum(stage wall)` to describe glue time between stages

Typical glue time sources include control flow between stages, await/resume gaps, scheduling gaps, and short helper work that is outside the chosen stage boundaries.

### 4. Separate total request timing from API preprocessing timing

The API server preprocess chain often ends before `response_start_ms`, `first_body_ms`, or overall `TTFT`. Report them separately.

## Recommended report structure

Include these sections:

1. Short conclusion
2. Ordered API server ASCII timeline
3. Stage wall breakdown
4. Concurrency window table
5. Total-time explanation
6. Hotspot function table
7. Why the times cannot be directly added
8. Optimization suggestions without source changes
9. Confirmation that the service has been stopped
