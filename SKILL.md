---
name: vllm-ascend-use
description: Explain vLLM and vllm-ascend architecture and guide practical Ascend NPU usage. Use when Codex needs to deploy a stock vLLM Ascend inference service, prepare cache-safe multimodal benchmark inputs, run the 10k-text-token plus 40-image single-concurrency benchmark, or compare baseline and modified services for output consistency.
---

# vLLM Ascend Use

## Overview

Use this skill as the default playbook for stock `vllm` plus `vllm-ascend` work on Ascend NPU. It covers four things:

1. Explain how vLLM V1 is structured and where `vllm-ascend` fits.
2. Deploy an OpenAI-compatible inference service with practical Ascend defaults such as HCCL tuning, eager mode, multimodal SHM cache, and CPU binding.
3. Generate and benchmark the single-concurrency multimodal case: `10k` text tokens plus `40` images of size `288x512`, with no repeated text or image content within or across requests.
4. Capture baseline outputs and compare later runs against them while keeping request order fixed and reducing avoidable randomness.

## Workflow

### 1. Build context first

Read the reference that matches the task:

- Read [references/architecture.md](references/architecture.md) when asked how `vllm` or `vllm-ascend` works.
- Read [references/deployment.md](references/deployment.md) when bringing up a service.
- Read [references/performance-testing.md](references/performance-testing.md) when running the `10k text token + 40 image` benchmark.
- Read [references/accuracy-testing.md](references/accuracy-testing.md) when validating output consistency.

### 2. Prefer stock serving paths

Prefer `vllm serve` instead of `python -m vllm.entrypoints.openai.api_server` unless you are debugging a very specific issue. This keeps the workflow aligned with upstream `vllm` and `vllm-ascend`.

### 3. Keep deployment and validation modes separate

Use two operating modes:

- Performance mode: prioritize the practical serving configuration, including multimodal SHM cache and the usual Ascend environment defaults.
- Accuracy mode: keep the same model and input set, but prefer the most stable configuration you can get, keep requests serial, keep order fixed, and make sampling parameters explicit.

Do not mix performance timing and accuracy comparison in the same run.

### 4. Use the bundled scripts

The scripts are thin utilities intended to reduce repeated manual work:

- [scripts/start_vllm_ascend_server.sh](scripts/start_vllm_ascend_server.sh)
  Start a stock `vllm serve` process with Ascend-friendly defaults.
- [scripts/generate_multimodal_dataset.py](scripts/generate_multimodal_dataset.py)
  Generate the `13`-round cache-safe multimodal dataset.
- [scripts/benchmark_chat_service.py](scripts/benchmark_chat_service.py)
  Run the single-concurrency benchmark and report `TTFT`, `TPOT`, and `E2E` statistics.
- [scripts/capture_chat_outputs.py](scripts/capture_chat_outputs.py)
  Capture serial output traces in fixed round order.
- [scripts/compare_text_outputs.py](scripts/compare_text_outputs.py)
  Compare two captured output files by round.

## Operating Rules

- Keep `vllm` and `vllm-ascend` versions matched. Treat version mismatch as the first thing to rule out.
- Assume multimodal local file access is blocked unless `--allowed-local-media-path` is set to the parent directory that contains the files.
- Use `--enforce-eager` for the controlled Ascend service flows in this skill. It is the safest baseline for debugging, profiling, and multimodal bring-up.
- For multimodal TP deployments, treat `--mm-processor-cache-type shm` and `--mm-encoder-tp-mode data` as deliberate configuration choices, not defaults you can silently drop.
- For accuracy comparison, keep the dataset order fixed, keep requests serial, keep `temperature=0`, and make `max_completion_tokens` explicit.
- Before blaming a modified build for output drift, run a baseline self-check first. If baseline vs baseline is already unstable, exact-match failures are not enough to prove a regression.

## Practical Prompts

- "Use this skill to explain the request path from API server to engine core to TP worker in vLLM Ascend."
- "Use this skill to deploy a Qwen multimodal service on Ascend with eager mode, SHM multimodal cache, and CPU binding."
- "Use this skill to generate the 13-round benchmark dataset and run the single-concurrency multimodal benchmark."
- "Use this skill to capture baseline outputs and compare them with a modified service while keeping request order fixed."
