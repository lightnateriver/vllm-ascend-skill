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

## Common false negatives

The most common non-capability failure is output truncation.

Symptoms:

- the answer clearly starts listing the correct sequence
- the final one or two shapes are missing
- `finish_reason` is `length`

Mitigation:

- constrain the prompt to return only a comma-separated list
- increase `max_completion_tokens` for multi-image and video cases

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
