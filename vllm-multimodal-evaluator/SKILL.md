---
name: vllm-multimodal-evaluator
description: Build simple shape-based multimodal test data and run two-phase capability checklists (format ingestion + semantic understanding) against a stock vLLM or vllm-ascend OpenAI-compatible service. Supports file://, Base64, and HTTP media modes. Reports include service config table (dtype, chunked-prefill, async-scheduling, prefix-caching, function calling). Use when you need to verify format-level media ingestion separately from model-level content understanding across three transport modes.
---

# vLLM Multimodal Evaluator

## Overview

Use this skill when the job is not generic deployment or large-scale benchmarking, but targeted multimodal capability evaluation for a stock `vllm` or `vllm-ascend` OpenAI-compatible service.

The evaluation uses a **two-phase** approach:
- **Phase 1 — Format ingestion**: Verify the service can read each media format (HTTP 200 + no error). Tests all formats across `file://`, Base64, and HTTP modes.
- **Phase 2 — Semantic understanding**: Verify the model correctly understands content (keyword matching against expected groups). Same format × mode matrix as Phase 1.

This separation means Phase 2 FAIL results can be confidently attributed to **model capability limits** rather than transport or format-handling issues.

Treat this as a companion child skill of `vllm-ascend-use`:

- Use `vllm-ascend-use` for general architecture, service bring-up, benchmark generation, and pre-LLM input consistency checks.
- Use `vllm-multimodal-evaluator` when you need deterministic local image or video fixtures and a concrete multimodal support checklist.

## Workflow

### 1. Confirm the evaluation target

Before running the checklist, confirm all of these:

- the target service exposes `/v1/models` and `/v1/chat/completions`
- the target model is multimodal, for example `Qwen3.5-4B`
- the service was started with `--allowed-local-media-path` covering the parent directory that holds `pics/` and `video/`

If you still need the service deployment workflow, read the parent skill `vllm-ascend-use` and its [references/deployment.md](../vllm-ascend-use/references/deployment.md) first.

### 2. Generate the local test dataset

Use the bundled scripts to build a deterministic test set made of simple blue shapes on a green background.

- Run [scripts/generate_shape_dataset.py](scripts/generate_shape_dataset.py) to create the image set.
- Run [scripts/generate_shape_videos.py](scripts/generate_shape_videos.py) after the JPG images exist to build the video set.

Read [references/dataset-layout.md](references/dataset-layout.md) before changing resolutions, formats, or naming rules.

The default dataset shape is:

- images under `pics/<resolution>/<format>/<shape>.<ext>`
- videos under `video/<resolution>/<format>/shapes.<ext>`

### 3. Start the service for local media evaluation

If the user wants you to start the model service as part of the evaluation, use [scripts/start_qwen35_4b_vllm.sh](scripts/start_qwen35_4b_vllm.sh).

- A generic vLLM startup script that works on both NVIDIA GPU and Ascend NPU.
- Default enables: chunked-prefill, async-scheduling, prefix-caching, function calling, local media.
- Set `MODEL_PATH` explicitly. See script header for all supported env vars (ENABLE_CHUNKED_PREFILL, ENABLE_ASYNC_SCHEDULING, ENABLE_PREFIX_CACHING, ENABLE_FUNCTION_CALLING etc. can be set to 0 to disable).
- Example:
  ```bash
  MODEL_PATH=/path/to/Qwen3.5-4B \
  ALLOWED_LOCAL_MEDIA_PATH=/path/to/project \
  PORT=8000 \
  bash scripts/start_qwen35_4b_vllm.sh
  ```
- If testing HTTP media mode, also start a static file server:
  ```bash
  python3 -m http.server 9000 --directory /path/to/project
  ```
- Ascend NPU users should source their ascend-toolkit environment before running.

### 3.5 HTTP Static Server Validation

- When HTTP mode is in scope, verify one image URL and one video URL directly before the capability run.
- If HTTP-only failures appear while `file_url` or Base64 pass, treat the issue as static-file serving or transport first.
- Keep the static media root and direct validation command in the final report.

### 4. Run the multimodal checklist

Use [scripts/run_multimodal_capability_tests.py](scripts/run_multimodal_capability_tests.py) to send the capability matrix to the target service and write a machine-readable plus human-readable report.

The checklist runs in two phases:
1. **Format ingestion tests** (`INGEST-*` cases): quick checks (16 tokens) that verify the service can read each media format across all transport modes. Only checks HTTP 200 + no error — no semantic validation.
2. **Semantic understanding tests** (all other cases): full 512-token checks that verify the model correctly describes shapes, colors, and sequences.

The checklist uses `max_completion_tokens=512` by default for every semantic case. Keep this default unless the user explicitly asks for a different output-token budget.

#### CLI arguments

The script accepts service configuration flags that are displayed in the report header:

| Flag | Default | Description |
|------|---------|-------------|
| `--dtype` | `bfloat16` | Model precision (bfloat16/float16/float32) |
| `--chunked-prefill` | `True` | Enable chunked prefill |
| `--async-scheduling` | `True` | Enable async scheduling |
| `--prefix-caching` | `True` | Enable prefix caching |
| `--function-calling` | `True` | Enable function calling (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`) |
| `--gpu-memory-utilization` | `0.7` | GPU memory utilization ratio |
| `--enforce-eager` | `True` | Enable eager mode |
| `--media-base-url` | `None` | Base URL for HTTP mode tests, e.g. `http://127.0.0.1:9000`. When set, HTTP-mode variants of all semantic and ingestion cases are added automatically. |

#### Three media transport modes

When `--media-base-url` is set, each capability is tested across all three transport modes:

| Mode | URL format | Requirements |
|------|-----------|-------------|
| `file_url` | `file:///path/to/file.jpg` | Service needs `--allowed-local-media-path` |
| `base64` | `data:image/jpeg;base64,...` | No extra configuration |
| `http` | `http://host:port/path/to/file.jpg` | Static file server + `--media-base-url` |

Example with full config and HTTP mode:

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B \
  --dtype bfloat16 \
  --chunked-prefill True \
  --async-scheduling True \
  --prefix-caching True \
  --function-calling True \
  --media-base-url http://127.0.0.1:9000
```

#### Report structure

The Markdown report contains five sections:

1. **服务配置** — Table of service flags (dtype, chunked-prefill, async-scheduling, prefix-caching, function calling, media-base-url, etc.)
2. **媒体格式读取测试** — Phase 1 ingestion results (per-format × per-mode table + summary)
3. **语义理解测试** — Phase 2 semantic results (per-category tables)
4. **语义理解汇总** — Summary matrix of all semantic categories
5. **失败 Case 明细** — Failures with HTTP status and output excerpt
6. **完整 Case 输入与输出** — Full request payloads and model outputs for reproduction

The script covers these capability categories (with HTTP variants when `--media-base-url` is set):

- image format support through local `file://` URLs
- image format support through Base64 data URLs
- image format support through HTTP URLs
- image resolution support (file:// and HTTP)
- seven-image understanding (file://, Base64, HTTP)
- interleaved text and image content ordering
- video format support (file:// and HTTP)
- video resolution support (file:// and HTTP)
- video first-shape, last-shape, and ordered-sequence checks (file:// and HTTP)

### 5. Run the Function Calling checklist

Use [scripts/fc_test.py](scripts/fc_test.py) against a service started with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`.

The script accepts:
- `--endpoint` (default: `http://127.0.0.1:8000/v1/chat/completions`)
- `--model` (default: model path)
- `--test-file` (default: auto-detect from script location)

Example:
```bash
python scripts/fc_test.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model /path/to/Qwen3.5-4B
```

The script reads [scripts/function_calling_test.json](scripts/function_calling_test.json) which contains 15 test cases covering:

- single function call with required parameters
- parallel tool calls
- missing-parameter scenarios (small models may fill defaults instead of asking)
- multi-turn context-dependent calls
- no-call scenarios (casual chat)
- fuzzy/invalid parameter handling
- long-text interference extraction

Evaluation is relaxed: only check function name match and required parameter existence, not exact string values.

Read [references/checklist-design.md](references/checklist-design.md) when you need to adjust prompts, expected groups, or PASS or FAIL interpretation.

### 6. Interpret failures carefully

Do not treat every FAIL as an unsupported media type. The two-phase design helps here:

**Phase 1 (ingestion) FAIL** → Service cannot read the media file. Check:
- `--allowed-local-media-path` is set correctly
- Static file server is running (for HTTP mode)
- File permissions and paths are correct
- vLLM version supports the format

**Phase 2 (semantic) FAIL** → Service can read the file but the model gave a wrong answer.
1. request construction or transport problems (unlikely if ingestion PASSed)
2. output truncation due to weak prompt control or low `max_completion_tokens`
3. genuine model understanding errors

The default script already uses stronger prompts and larger `max_completion_tokens` for multi-image and video cases to reduce false negatives from verbose reasoning.

### 6.5 Capability Root-Cause Policy

- Ingestion failures are engineering or pipeline candidates first.
- Semantic failures after successful ingestion are model capability candidates first, unless timeout or truncation dominates the behavior.
- Output that does not collapse to the required short format should be reported as a protocol or formatting issue, not blindly counted as a vision failure.
- If a historical video or HTTP failure is later fixed by repairing evaluator transport handling, static media serving, or timeout policy, remove that historical case from the model error count.

### 7. Keep reports reproducible

When reporting results, include:

- model path or served model id
- service base URL
- the dataset root used by the run
- the generated report paths
- service config flags (dtype, chunked-prefill, async-scheduling, etc.)
- media transport modes tested (file_url, base64, http)
- PASS or FAIL counts by category (separated for ingestion and semantic)
- any residual failures that look like real capability gaps

The Markdown report must keep enough information for reproduction and debugging:

- summary tables by capability category
- the output token limit used by each case
- the full `/v1/chat/completions` request payload for every case, including text, media URL or Base64 data, and `max_completion_tokens`
- the full model output for every case

The JSON report stores the same request payload and full model output in machine-readable form.

### 7.5 Standard Capability Summary

- In addition to the detailed report, keep a compact summary for downstream acceptance reporting.
- The compact summary should explicitly separate:
  - `engineering_errors`
  - `model_limitations`
  - `not_counted_items`
  - `counts_by_status`
  - `counts_by_test_type`
  - `artifact_paths`

## Operating Rules

- Prefer deterministic synthetic fixtures over scraped or user-supplied media when the task is capability validation.
- Keep the shape order fixed as `square, rectangle, rhombus, circle, triangle, cylinder, cube`.
- Keep single-image prompts short, but constrain multi-image and video prompts to return only comma-separated lists or single labels.
- Keep ingestion tests at 16 tokens (just enough to get a response, no semantic validation).
- Keep checklist output tokens at 512 for semantic tests by default. If a case still fails with `finish_reason=length` or a visibly truncated answer, report that separately instead of lowering the token budget.
- For local media tests, treat missing `--allowed-local-media-path` as the first thing to rule out.
- When a multi-image or video answer is close but incomplete, check whether the response was truncated before concluding the capability failed.
- When the target service is already running, do not restart it unless the user asks or the current configuration blocks local media tests.
- When comparing results across runs, ensure the same service config and media modes are used.

## Resources

- [scripts/generate_shape_dataset.py](scripts/generate_shape_dataset.py)
  Generate shape images across multiple formats and resolutions.
- [scripts/generate_shape_videos.py](scripts/generate_shape_videos.py)
  Turn the JPG fixtures into low-size multi-format videos.
- [scripts/start_qwen35_4b_vllm.sh](scripts/start_qwen35_4b_vllm.sh)
  Start a vLLM service (NVIDIA GPU / Ascend NPU) with chunked prefill, async scheduling, prefix caching, function calling, and local media enabled by default.
- [scripts/run_multimodal_capability_tests.py](scripts/run_multimodal_capability_tests.py)
  Run the two-phase checklist (format ingestion + semantic understanding) and emit Markdown plus JSON reports. Supports file_url, base64, and HTTP media modes. Includes service config in report header.
- [scripts/function_calling_test.json](scripts/function_calling_test.json)
  15 standard test cases for Function Calling evaluation.
- [scripts/fc_test.py](scripts/fc_test.py)
  Run the Function Calling test suite with configurable endpoint/model. Outputs PASS/FAIL per case.
- [references/dataset-layout.md](references/dataset-layout.md)
  Define the fixture directory layout, naming rules, and media properties.
- [references/checklist-design.md](references/checklist-design.md)
  Define the capability matrix and failure interpretation rules.

## Practical Prompts

- "Use $vllm-multimodal-evaluator to generate the synthetic image and video fixtures for multimodal testing."
- "Use $vllm-multimodal-evaluator to start Qwen3.5-4B with local media enabled and run the multimodal checklist."
- "Use $vllm-multimodal-evaluator to verify whether this service supports JPG, PNG, WebP, BMP, and TIFF through file URL, Base64, and HTTP."
- "Use $vllm-multimodal-evaluator to test multi-image ordering, interleaved text plus image messages, and video sequence understanding across all three media modes."
- "Use $vllm-multimodal-evaluator to run the two-phase evaluation: first check format ingestion, then check semantic understanding."
- "Use $vllm-multimodal-evaluator to rerun the checklist and summarize which failures are format ingestion issues versus real model understanding gaps."
- "Use $vllm-multimodal-evaluator to run the Function Calling test suite against a service with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`."
- "Use $vllm-multimodal-evaluator with --media-base-url to include HTTP transport mode in all tests."
