# Metric Definitions

## Timing terms

- `HTTP total wall time`
  End-to-end wall time for the profiled HTTP request on the API server side.
- `API server CPU time`
  CPU time consumed by the API server process during the profiled request window.
- `API server preprocess chain`
  The custom request span from `HfRenderer.render_messages_async` start to the final `SingleWriterShmObjectStorage.copy_to_buffer` end.

## Function timing labels

- `wall-critical-path`
  The elapsed wall span on the request timeline that blocks forward progress for the ordered step being discussed.
- `aggregate-total`
  The sum of all invocations for the same function. This may exceed wall time when many calls overlap.
- `inclusive-total`
  The elapsed time of one function invocation including its child calls. Do not add this directly to child totals.

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

### 3. Separate total request timing from API preprocessing timing

The API server preprocess chain often ends before `response_start_ms`, `first_body_ms`, or overall `TTFT`. Report them separately.

## Recommended report structure

Include these sections:

1. Short conclusion
2. Ordered API server ASCII timeline
3. Total-time explanation
4. Hotspot function table
5. Why the times cannot be directly added
6. Optimization suggestions without source changes
7. Confirmation that the service has been stopped
