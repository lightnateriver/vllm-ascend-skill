---
name: vllm-multimodal-evaluator
description: Build simple shape-based multimodal test data and run capability checklists against a stock vLLM or vllm-ascend OpenAI-compatible service. Use when Codex needs to generate deterministic image and video inputs, deploy a Qwen3.5 multimodal service with local media enabled, verify image format support through file URL and Base64 requests, test multi-image and interleaved text-plus-image messages, or validate video format, resolution, and sequence understanding with a reusable PASS or FAIL report.
---

# vLLM Multimodal Evaluator

## Overview

Use this skill when the job is not generic deployment or large-scale benchmarking, but targeted multimodal capability evaluation for a stock `vllm` or `vllm-ascend` OpenAI-compatible service.

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

If the user wants you to start the model service as part of the evaluation, use [scripts/start_qwen35_4b_vllm_ascend.sh](scripts/start_qwen35_4b_vllm_ascend.sh).

- The wrapper is intentionally opinionated for the tested `Qwen3.5-4B` multimodal flow.
- It enables eager mode, TP, SHM multimodal cache, data-style multimodal encoder TP, and CPU binding.
- Set `MODEL_PATH` explicitly.
- Set `ALLOWED_LOCAL_MEDIA_PATH` to the project root that contains `pics/` and `video/`.

### 4. Run the multimodal checklist

Use [scripts/run_multimodal_capability_tests.py](scripts/run_multimodal_capability_tests.py) to send the capability matrix to the target service and write a machine-readable plus human-readable report.

The checklist uses `max_completion_tokens=512` by default for every case. Keep this default unless the user explicitly asks for a different output-token budget.

The script covers:

- image format support through local `file://` URLs
- image format support through Base64 data URLs
- image resolution support
- seven-image understanding with file URLs and Base64
- interleaved text and image content ordering
- video format support
- video resolution support
- video first-shape, last-shape, and ordered-sequence checks

Read [references/checklist-design.md](references/checklist-design.md) when you need to adjust prompts, expected groups, or PASS or FAIL interpretation.

### 5. Interpret failures carefully

Do not treat every FAIL as an unsupported media type.

Separate failures into:

1. request construction or transport problems
2. service-side media ingestion problems
3. output truncation due to weak prompt control or low `max_completion_tokens`
4. genuine model understanding errors

The default script already uses stronger prompts and larger `max_completion_tokens` for multi-image and video cases to reduce false negatives from verbose reasoning.

### 6. Keep reports reproducible

When reporting results, include:

- model path or served model id
- service base URL
- the dataset root used by the run
- the generated report paths
- PASS or FAIL counts by category
- any residual failures that look like real capability gaps

The Markdown report must keep enough information for reproduction and debugging:

- summary tables by capability category
- the output token limit used by each case
- the full `/v1/chat/completions` request payload for every case, including text, media URL or Base64 data, and `max_completion_tokens`
- the full model output for every case

The JSON report stores the same request payload and full model output in machine-readable form.

## Operating Rules

- Prefer deterministic synthetic fixtures over scraped or user-supplied media when the task is capability validation.
- Keep the shape order fixed as `square, rectangle, rhombus, circle, triangle, cylinder, cube`.
- Keep single-image prompts short, but constrain multi-image and video prompts to return only comma-separated lists or single labels.
- Keep checklist output tokens at 512 by default. If a case still fails with `finish_reason=length` or a visibly truncated answer, report that separately instead of lowering the token budget.
- For local media tests, treat missing `--allowed-local-media-path` as the first thing to rule out.
- When a multi-image or video answer is close but incomplete, check whether the response was truncated before concluding the capability failed.
- When the target service is already running, do not restart it unless the user asks or the current configuration blocks local media tests.

## Resources

- [scripts/generate_shape_dataset.py](scripts/generate_shape_dataset.py)
  Generate shape images across multiple formats and resolutions.
- [scripts/generate_shape_videos.py](scripts/generate_shape_videos.py)
  Turn the JPG fixtures into low-size multi-format videos.
- [scripts/start_qwen35_4b_vllm_ascend.sh](scripts/start_qwen35_4b_vllm_ascend.sh)
  Start a stock `vllm serve` flow for `Qwen3.5-4B` on Ascend with local media enabled.
- [scripts/run_multimodal_capability_tests.py](scripts/run_multimodal_capability_tests.py)
  Run the checklist and emit Markdown plus JSON reports.
- [references/dataset-layout.md](references/dataset-layout.md)
  Define the fixture directory layout, naming rules, and media properties.
- [references/checklist-design.md](references/checklist-design.md)
  Define the capability matrix and failure interpretation rules.

## Practical Prompts

- "Use $vllm-multimodal-evaluator to generate the synthetic image and video fixtures for multimodal testing."
- "Use $vllm-multimodal-evaluator to start Qwen3.5-4B with local media enabled and run the multimodal checklist."
- "Use $vllm-multimodal-evaluator to verify whether this service supports JPG, PNG, WebP, BMP, and TIFF through file URL and Base64."
- "Use $vllm-multimodal-evaluator to test multi-image ordering, interleaved text plus image messages, and video sequence understanding."
- "Use $vllm-multimodal-evaluator to rerun the checklist and summarize which failures are transport issues versus real model understanding gaps."
