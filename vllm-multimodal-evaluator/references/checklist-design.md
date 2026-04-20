# Checklist Design

Use this reference when adjusting the capability checklist or interpreting test results.

## Coverage matrix

The bundled script currently covers these categories:

1. image single-file format support through local file URLs
2. image single-file format support through Base64 data URLs
3. image resolution support
4. seven-image ordering with file URLs
5. seven-image ordering with Base64
6. interleaved text plus image ordering
7. video format support
8. video resolution support
9. video first-shape, last-shape, and ordered-sequence checks

## PASS or FAIL logic

Each case uses expected keyword groups. A case passes when at least one synonym from every expected group is present in the model output.

Examples:

- single image cases expect:
  - shape synonym
  - blue synonym
  - green synonym
- full-sequence cases expect:
  - one synonym group per shape, in the fixed order

## Output token budget

The checklist uses `max_completion_tokens=512` by default for every request. This includes simple single-image cases, multi-image cases, interleaved text-plus-image cases, and video cases.

Keep this default token budget unless a user explicitly asks for a different value. If a response is still truncated at 512 tokens, classify that as a prompt or generation-limit issue rather than a media-ingestion failure.

## Common false negatives

The most common non-capability failure is output truncation.

Symptoms:

- the answer clearly starts listing the correct sequence
- the final one or two shapes are missing
- `finish_reason` is `length`

Mitigation:

- constrain the prompt to return only a comma-separated list
- keep `max_completion_tokens` at 512 by default, and only increase it when a specific run still shows `finish_reason=length` or visibly truncated output

## Real capability gaps

Treat a failure as a likely model capability gap only after ruling out:

- local media path access problems
- wrong MIME or URL construction
- timeout or transport failures
- output truncation

## Reporting guidance

Summaries should separate:

- media ingestion support
- multi-image ordering support
- interleaved content support
- video sequence understanding support

This distinction matters because a model may support a media format while still failing a harder reasoning question on top of that media.

The Markdown report should include full reproduction detail for each case:

- the full `/v1/chat/completions` request payload, including text content, media references or Base64 data, and `max_completion_tokens`
- the full model output, not a truncated preview
- the per-case output token limit shown in summary tables

The JSON report should keep the same request payload and full output in structured fields.
