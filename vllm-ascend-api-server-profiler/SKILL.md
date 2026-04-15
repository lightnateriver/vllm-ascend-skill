---
name: vllm-ascend-api-server-profiler
description: Profile and analyze stock vllm-ascend OpenAI API server hotspots on Ascend NPU with external monkey patch only. Use when Codex needs to measure API server-side wall time and function hotspots for Qwen or Qwen3.5 multimodal requests, collect a request-scoped span from HfRenderer.render_messages_async to copy_to_buffer, generate ordered API server timelines and hotspot tables, or explain API server bottlenecks without treating engine core or TP worker as the conclusion.
---

# vLLM Ascend API Server Profiler

## Overview

Use this skill as the focused playbook for stock `vllm` plus `vllm-ascend` API server profiling on Ascend. Keep the work scoped to API server request preprocessing and request handling. Do not treat engine core or TP worker execution as the main conclusion.

## Workflow

### 1. Confirm the scope

Apply this skill only when the goal is API server bottleneck analysis.

- Keep `vllm` and `vllm-ascend` source trees unchanged.
- Do not use `0407` patches.
- Do not use phase-based profiling schemes.
- Implement all request tracing and function timing through external monkey patch only.
- Report API server totals and function hotspots separately from engine-side execution.

### 2. Prepare the input set

Use the companion `vllm-ascend-use` skill when you need to generate the standard multimodal benchmark dataset.

- For the stock Qwen3.5 case, use `10k` text tokens plus `40` images of size `288x512`.
- For `3 warmup + 1 profile`, keep all text and image data unique within and across rounds.
- Set `--allowed-local-media-path` to the parent directory that contains the generated images.

Read [references/qwen35-stock-config.md](references/qwen35-stock-config.md) for the default Ascend environment, serving arguments, and target function list.

### 3. Start the profiled API server

Use [scripts/start_profiled_api_server.sh](scripts/start_profiled_api_server.sh) to launch a stock `vllm serve` flow through the external profiler entrypoint.

- The wrapper sources the Ascend environment and exports the profiling headers and output directory defaults.
- Keep the profiler attached to the whole API server request.
- Record the outer API server span from `HfRenderer.render_messages_async` start to the last `SingleWriterShmObjectStorage.copy_to_buffer` end.

The wrapper calls [scripts/run_vllm_api_server_profiled.py](scripts/run_vllm_api_server_profiled.py), which holds the external monkey patch logic.

### 4. Drive the profile round

Use [scripts/run_profiled_dataset_rounds.py](scripts/run_profiled_dataset_rounds.py) to send `3` warmup rounds and `1` profiled round.

- Profile exactly one request by setting the profiling header on the profile round only.
- Keep `stream=true` so `TTFT`, `TPOT`, and `E2E` stay visible.
- Stop the service after collection to release NPU resources.

### 5. Summarize the artifacts

Use [scripts/summarize_api_server_profile.py](scripts/summarize_api_server_profile.py) on the generated `functions.json` and the driver results JSON.

The summary must preserve three different timing views:

- `HTTP total wall time`
- `API server CPU time`
- `API server preprocess chain`

Read [references/metric-definitions.md](references/metric-definitions.md) before writing conclusions. Do not add together times that overlap by nesting or concurrency.

### 6. Keep the hotspot report actionable

Prefer functionally meaningful steps over broad wrappers.

- Good targets: `ProcessorInputs.get_mm_hashes`, `Qwen2VLImageProcessorFast._preprocess`, `Qwen2TokenizerFast.__call__`, `SingleWriterShmObjectStorage.copy_to_buffer`.
- Useful context wrappers: `InputProcessingContext.call_hf_processor`, `Qwen3VLProcessor.__call__`.
- Poor final conclusions: `create_chat_completion` or similarly broad wrappers that mix unrelated work.

## Reporting Rules

- Draw the API server timeline in chronological order, not only as nested blocks.
- Mark which numbers are `wall-critical-path`, `aggregate-total`, and `inclusive-total`.
- Include a hotspot table with `function`, `calls`, `total_ms`, and `avg_ms`.
- Explain why the times cannot be directly summed.
- Give optimization suggestions only under the constraint that `vllm` and `vllm-ascend` source must remain unchanged.
- Stop the service after the run if it is still alive.

## Resources

- [scripts/start_profiled_api_server.sh](scripts/start_profiled_api_server.sh)
  Start the stock profiled API server with the standard Ascend environment and profiling defaults.
- [scripts/run_vllm_api_server_profiled.py](scripts/run_vllm_api_server_profiled.py)
  External profiler entrypoint that installs request-scoped monkey patches and writes profile artifacts.
- [scripts/run_profiled_dataset_rounds.py](scripts/run_profiled_dataset_rounds.py)
  Send `3 warmup + 1 profile` rounds while applying the profile header to the target round only.
- [scripts/summarize_api_server_profile.py](scripts/summarize_api_server_profile.py)
  Convert `functions.json` and driver results into a timeline, hotspot table, and analysis scaffold.
- [references/metric-definitions.md](references/metric-definitions.md)
  Define timing terms and interpretation rules.
- [references/qwen35-stock-config.md](references/qwen35-stock-config.md)
  Capture the stock Qwen3.5 API server profiling baseline.

## Practical Prompts

- "Use $vllm-ascend-api-server-profiler to run a stock Qwen3.5 API server profile and summarize API-side hotspots."
- "Use $vllm-ascend-api-server-profiler to explain why `load_from_url_async` aggregate time does not equal wall time."
- "Use $vllm-ascend-api-server-profiler to produce an ordered API server timeline and hotspot table from these profile artifacts."
