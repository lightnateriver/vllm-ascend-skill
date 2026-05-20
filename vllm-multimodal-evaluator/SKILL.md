---
name: vllm-multimodal-evaluator
description: Build deterministic multimodal fixtures and run two-phase capability checks (format ingestion + semantic understanding) against local vLLM or vllm-ascend services. Standard runs default to a selective three-transport matrix for key Phase 2 image understanding items, with other evaluator items on local_path unless full transport coverage is explicitly requested.
---

# vLLM Multimodal Evaluator

## What this skill is for

Use this skill when you need to answer a simple but important question:

> Can the service ingest multimodal inputs correctly, and can the model understand them at a basic capability level?

This skill is for capability validation, not benchmark scoring.

It is especially useful for:

- checking whether image/video input paths work
- checking whether `local_path`, `base64`, and `http` behave consistently
- separating pipeline issues from model capability issues
- validating large-image smoke support
- validating 1 to 10 multi-video understanding
- validating function calling support

## What it does not cover

Do not use this skill for:

- `L0`
- `L0.5`
- `MME`
- `MMBench`
- long-form accuracy benchmarking

Those belong to `vllm-multimodal-precision-testing`.

## Core design

This skill uses a two-phase model:

1. `Phase 1 - Format ingestion`
   Confirm the service can actually read the media.
2. `Phase 2 - Semantic understanding`
   Confirm the model can answer correctly after ingestion succeeds.

This split matters because it lets you classify failures more cleanly:

- `Phase 1` failures usually mean engineering or serving problems
- `Phase 2` failures after successful ingestion usually mean model capability gaps
- terse output drift or explanation-heavy answers are output-format/protocol issues

## Transport naming
If the user directly runs the checklist without prebuilt fixtures, the bundled checklist script now auto-generates missing `pics/` and `video/` before evaluating.

Externally, this skill uses:

- `local_path`
- `base64`
- `http`

Internally, `local_path` is still implemented as a `file://` URL.

Reports keep both:

- `transport_mode=local_path`
- `transport_impl=file_url`

## Default capability coverage
The bundled script also queries `/v1/models` first and normalizes the requested `--model` to the actual served model id when needed, for example when the service exposes a trailing slash in the model id.

Standard capability runs default to:

- selective three-transport coverage for key Phase 2 image understanding items
- `local_path` coverage for the remaining evaluator items
- standard image and video ingestion suites
- standard image and video semantic suites
- 1 to 10 multi-video semantic suites
- large-image smoke
- function calling

### Default suites

| Suite | Default | Purpose |
| --- | :---: | --- |
| `phase1_ingestion_standard` | On | Verify standard image/video format ingestion |
| `phase2_semantic_standard` | On | Verify standard image/video semantic understanding |
| `phase2_semantic_multi_video_standard` | On | Verify 1 to 10 multi-video semantic understanding |
| `phase1_ingestion_large_image_smoke` | On | Verify 4K-class image ingestion |
| `phase2_semantic_large_image_smoke` | On | Verify 4K-class image semantics |
| `phase2_function_calling_standard` | On | Verify function calling ability and output chain |

### Default transport matrix

| Transport | On | Notes |
| --- | :---: | --- |
| `local_path` | Yes | Default for all evaluator items; requires `--allowed-local-media-path` on the served model |
| `base64` | Yes | Enabled by default only for selected Phase 2 image understanding items |
| `http` | Yes | Enabled by default only for selected Phase 2 image understanding items; requires local static media server |

By default, only these Phase 2 items use `local_path`, `base64`, and `http` together:

- single-image semantic understanding
- multi-image understanding
- interleaved text/image ordering

Other evaluator items default to `local_path` only.

If the user explicitly asks for full transport coverage on every evaluator item, run with:

```bash
python scripts/run_multimodal_capability_tests.py --full-transport-matrix
```

## Supported test content

### Phase 1

- image format ingestion
- image resolution ingestion
- video format ingestion
- video resolution ingestion

### Phase 2

- single-image semantic understanding
- image resolution semantic understanding
- multi-image understanding
- interleaved text/image ordering
- video semantic understanding
- video order and detail understanding
- multi-video order understanding

### Extended smoke

Default large-image smoke resolves to:

- `4096x4096`
- `4096x6144`
- `4096x8192`

Multi-video standard runs use single-shape clips stored under `video/720x1280/mp4/<shape>.mp4`.

## Run flow

Suggested flow:

1. generate or confirm fixtures
2. confirm the target service is reachable through `/v1/models` and `/v1/chat/completions`
3. run the standard capability checklist
4. inspect phase-separated failures
5. only then move to precision testing

## Failure attribution

Use the following rule of thumb:

- `Phase 1 FAIL`
  First suspect media access, static server, MIME/path issues, permissions, or service startup issues.
- `Phase 2 FAIL` after successful ingestion
  First suspect model capability.
- `unknown` or explanation-heavy output
  First suspect output protocol or extraction instability.

Do not keep counting a known transport problem as a model error after the transport issue has been fixed and rerun.

## Report fields

Capability reports should include at least:

- service configuration
- selected transport modes
- whether full transport matrix was enabled
- enabled optional suites
- included test items
- counts by suite
- counts by media scale
- counts by transport mode
- `failure_class`
- `root_cause_note`

## Current validated behavior

For the current repository version:

- standard ingestion coverage passes across `local_path`, `base64`, and `http`
- standard semantic coverage is broadly healthy across all three transport modes
- multi-video understanding is transport-consistent but model-limited and unstable
- large-image smoke ingestion passes, while large-image semantic smoke remains a model-capability concern

## Default command

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B \
  --media-base-url http://127.0.0.1:9000 \
  --auto-start-media-server
```

## Human summary

When you summarize results to a user, always say:

- what was tested
- which transports were tested
- whether large-image smoke and function calling were included
- how many cases passed or failed
- whether failures are engineering, protocol, or model-capability related
