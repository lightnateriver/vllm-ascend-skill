---
name: vllm-ascend-use
description: Explain vLLM and vllm-ascend architecture and guide practical Ascend NPU usage. Use when Codex needs to deploy a stock vLLM Ascend inference service, prepare cache-safe multimodal benchmark inputs, run the 10k-text-token plus 40-image single-concurrency benchmark, or compare baseline and modified services by checking whether the inputs immediately before the LLM are consistent.
---

# vLLM Ascend Use

## Overview

Use this skill as the default playbook for stock `vllm` plus `vllm-ascend` work on Ascend NPU. It covers four things:

1. Explain how vLLM V1 is structured and where `vllm-ascend` fits.
2. Deploy an OpenAI-compatible inference service with practical Ascend defaults such as HCCL tuning, eager mode, multimodal SHM cache, and CPU binding.
3. Generate and benchmark the single-concurrency multimodal case: `10k` text tokens plus `40` images of size `288x512`, with no repeated text or image content within or across requests.
4. Compare baseline and candidate runs by checking whether the inputs immediately before the LLM are consistent while keeping request order fixed and reducing avoidable randomness.

## Workflow

### 1. Build context first

Read the reference that matches the task:

- Read [references/architecture.md](references/architecture.md) when asked how `vllm` or `vllm-ascend` works.
- Read [references/deployment.md](references/deployment.md) when bringing up a service.
- Read [references/performance-testing.md](references/performance-testing.md) when running the `10k text token + 40 image` benchmark.
- Read [references/accuracy-testing.md](references/accuracy-testing.md) when validating pre-LLM input consistency.

If the request is specifically about API server hotspot profiling, use the companion skill `vllm-ascend-api-server-profiler` instead of overloading this general skill.
If the request is specifically about building simple local multimodal fixtures and running a capability support checklist, use the companion child skill `vllm-multimodal-evaluator`.

### 2. Prefer stock serving paths

Prefer `vllm serve` instead of `python -m vllm.entrypoints.openai.api_server` unless you are debugging a very specific issue. This keeps the workflow aligned with upstream `vllm` and `vllm-ascend`.

### 3. Keep deployment and validation modes separate

Use two operating modes:

- Performance mode: prioritize the practical serving configuration, including multimodal SHM cache and the usual Ascend environment defaults.
- Accuracy mode: keep the same model and input set, but prefer the most stable configuration you can get, keep requests serial, keep order fixed, make sampling parameters explicit, and compare the inputs immediately before the LLM instead of requiring the final decoded text to be identical.

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
  Send serial requests in fixed round order when you need a stable request driver for validation.
- [scripts/compare_text_outputs.py](scripts/compare_text_outputs.py)
  Legacy helper for text-level comparison. Do not treat exact output match as a required accuracy signal for this skill.
- Companion child skill: `vllm-multimodal-evaluator`
  Use it when the task shifts from service deployment or benchmark generation to synthetic image or video fixture creation and multimodal capability matrix testing.

## Operating Rules

- Keep `vllm` and `vllm-ascend` versions matched. Treat version mismatch as the first thing to rule out.
- Assume multimodal local file access is blocked unless `--allowed-local-media-path` is set to the parent directory that contains the files.
- Use `--enforce-eager` for the controlled Ascend service flows in this skill. It is the safest baseline for debugging, profiling, and multimodal bring-up.
- For multimodal TP deployments, treat `--mm-processor-cache-type shm` and `--mm-encoder-tp-mode data` as deliberate configuration choices, not defaults you can silently drop.
- For accuracy comparison, keep the dataset order fixed, keep requests serial, keep `temperature=0`, make `max_completion_tokens` explicit, and disable `chunked prefill` when the purpose is to compare the inputs immediately before the LLM.
- For this skill, the primary regression signal is whether the tensors and metadata immediately before the LLM are consistent between runs. Exact output-text match is optional and should not be the default pass or fail criterion.
- Before blaming a modified build for input drift, run a baseline self-check first. If baseline vs baseline is already unstable at the pre-LLM input level, later mismatches are not a clean regression signal yet.

## Practical Prompts

- "Use this skill to explain the request path from API server to engine core to TP worker in vLLM Ascend."
- "Use this skill to deploy a Qwen multimodal service on Ascend with eager mode, SHM multimodal cache, and CPU binding."
- "Use this skill to generate the 13-round benchmark dataset and run the single-concurrency multimodal benchmark."
- "Use this skill to compare whether two Ascend services receive the same inputs immediately before the LLM while keeping request order fixed."
