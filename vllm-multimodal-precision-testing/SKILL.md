---
name: vllm-multimodal-precision-testing
description: Run layered multimodal precision regression tests for local vLLM or vllm-ascend OpenAI-compatible services, especially Qwen3.5 and Qwen3-VL style models. Use when Codex needs to quickly smoke-test fixed image and video cases at L0, or run broader L1 benchmark checks with MME and MMBench after model tuning or deployment changes.
---

# vLLM Multimodal Precision Testing

## Overview

Use this skill to run the verified multimodal regression stack against a local OpenAI-compatible service.

- `L0`: fixed smoke suite for 7 images plus 3 video checks
- `L0.5`: deterministic `1` to `40` multi-image precision dataset
- `L1`: broader benchmark checks with `MME` and `MMBench_DEV_EN`

This skill is intended for repeatable post-tuning or post-deployment regression testing, not one-off demos.

## Preconditions

Use this skill only when the target service is already running and reachable through a local `/v1/chat/completions` endpoint.

Default assumptions used by the bundled scripts:

- host: `http://127.0.0.1:8000`
- model: `/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B`
- image dir: `assets/l0/pics/720x1280/jpg`
- video path: `assets/l0/video/720x1280/mp4/shapes.mp4`
- multi-pics dataset dir: `multi-pics-datasets/cases`
- MME TSV: `/tmp/MME.tsv`
- MMBench TSV: `/tmp/MMBench_DEV_EN.tsv`

All verified requests in this skill assume:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

If this flag is missing, short-answer and benchmark outputs may become verbose and unstable, which can invalidate extraction-based scoring.

## L0 Smoke

### Goal

Catch obvious regressions in:

- single-image recognition
- multi-image order handling
- prompt following for short constrained answers
- basic video understanding

### Run

```bash
python3 scripts/l0_multimodal_smoke.py
```

Use JSON output when needed:

```bash
python3 scripts/l0_multimodal_smoke.py --json
```

Override defaults only when necessary:

```bash
python3 scripts/l0_multimodal_smoke.py \
  --host http://127.0.0.1:8000 \
  --model /path/to/model \
  --image-dir /path/to/pics/720x1280/jpg \
  --video-path /path/to/video/720x1280/mp4/shapes.mp4
```

### Pass Rule

- `10/10` pass means the finalized L0 suite is healthy.
- Any failure should be treated as a real smoke regression until explained.

### Finalized Cases

- `img-1-single` -> `circle`
- `img-2-second` -> `cube`
- `img-3-order` -> `circle,cube,cylinder`
- `img-4-shape4` -> `rectangle`
- `img-5-circle-rhombus` -> `1,5`
- `img-6-shape6` -> `square`
- `img-7-last` -> `triangle`
- `video-first` -> `square`
- `video-last` -> `cube`
- `video-count` -> `7`

## L0.5 Multi-Pics

### Goal

Catch regressions that only appear when one request carries many images.

- single-request multi-image retrieval
- target index binding
- order sensitivity under image counts from `1` to `40`
- short constrained answering under heavier visual context

### Bundled Dataset

This skill now bundles a deterministic multi-image dataset:

- dataset root: `multi-pics-datasets/cases`
- generator: `multi-pics-datasets/generate_dataset.py`
- evaluator: `scripts/multi_pics_eval.py`

Dataset rules:

- case `01` contains `1` image and asks a strict `YES` or `NO` question
- cases `02` to `40` contain exactly `N` images and ask for exactly one target image index
- shape and color combinations are unique within one case
- a shape may repeat in one case, but repeated shapes always use different colors

### Run

Full run:

```bash
python3 scripts/multi_pics_eval.py \
  --wait-ready \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --dataset-dir multi-pics-datasets/cases \
  --json
```

Single case:

```bash
python3 scripts/multi_pics_eval.py \
  --wait-ready \
  --case 40 \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --dataset-dir multi-pics-datasets/cases \
  --json
```

### Readiness Policy

Do not start this evaluation only because HTTP is reachable.

The bundled evaluator supports `--wait-ready` and should be used by default. Readiness must require:

1. `/v1/models` returns HTTP `200`
2. a minimal text `/v1/chat/completions` request also returns HTTP `200`

This avoids false starts where the service is listening but still returns startup `502` responses.

### Result Interpretation

The evaluator reports:

- `correct`
- `wrong`
- `unknown`
- `timeout`
- `accuracy`

Artifacts are written under `multi-pics-runs/<run_name>/`:

- `01.json` to `40.json`
- `summary.json`
- `summary.csv`

Each case artifact keeps:

- question
- gold answer
- target image metadata
- image inventory
- raw model output
- extracted prediction
- final scoring status

## L1 Benchmarks

### Goal

Use `L1` after `L0` passes. The purpose is to track broader multimodal accuracy after each tuning round.

- `MME`: broad yes/no perception plus reasoning coverage, especially OCR, existence, count, translation, and calculation
- `MMBench_DEV_EN`: broad MCQ coverage across perception and reasoning, including localization, spatial reasoning, OCR, structure reading, and future prediction

Recommended use:

1. run `L0`
2. run `MME`
3. run `MMBench_DEV_EN`
4. compare against the previous tuned baseline

Do not use `L1` as the first signal when the service itself may be broken.

### L1 Attention Points

- Keep `enable_thinking=false` in benchmark requests.
- Prefer strict output constraints:
  `MME` should trend toward short yes/no answers.
  `MMBench` should be forced to output only one uppercase option letter.
- Judge by extraction, not by raw string equality to a verbose answer.
- Keep the prompt policy stable across runs, or historical scores will not be comparable.
- Keep model endpoint, prompt style, and concurrency stable when comparing tuning rounds.
- `MMBench_DEV_EN` contains circular variants and image references.
  Use the bundled script instead of ad hoc parsing.
- A small number of extraction fallbacks such as `Unknown` or `Z` may still occur.
  Track them as a separate metric because they often indicate prompt-following drift rather than pure vision failure.

## One-Command Run

Use the unified runner when you want the standard regression order in one command:

```bash
python3 scripts/run_full_regression.py
```

Useful overrides:

```bash
python3 scripts/run_full_regression.py \
  --host http://127.0.0.1:8000 \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B \
  --mme-tsv /tmp/MME.tsv \
  --mmbench-tsv /tmp/MMBench_DEV_EN.tsv \
  --concurrency 8 \
  --json
```

The unified runner:

1. runs `L0`
2. runs `MME`
3. runs `MMBench_DEV_EN`
4. prints a final JSON summary with per-step pass status
5. auto-downloads missing `MME` and `MMBench_DEV_EN` TSV files by default

Optional flags:

- `--skip-l0`
- `--skip-mme`
- `--skip-mmbench`
- `--no-auto-download`

Output artifacts are preserved by the underlying scripts:

- `L0`
  JSON output includes every fixed case and its returned content.
- `L0.5`
  `multi-pics-runs/<run_name>/summary.json` keeps the full run summary and all case details.
  `multi-pics-runs/<run_name>/<case>.json` keeps each case question, gold answer, prediction, extracted result, and status.
  `multi-pics-runs/<run_name>/summary.csv` keeps one scored row per case.
- `MME`
  `*.pred.tsv` keeps one row per question with `question`, `answer`, `prediction`, `extracted`, and `score`.
- `MMBench`
  `*.pred_all.tsv` keeps every raw row and variant with `question`, `answer`, `prediction`, `extracted`, and `row_hit`.
  `*.pred.tsv` keeps the scored main rows with grouped `hit`.

## L1 MME

### Dataset Objective

`MME` is the first L1 benchmark in this skill because it is easy to interpret and broad enough to catch regressions in:

- OCR
- counting
- existence
- position
- posters and scene understanding
- commonsense reasoning
- numerical calculation
- text translation

### Preparation

Download the official TSV if missing:

```bash
curl -k -L --fail --max-time 120 -o /tmp/MME.tsv \
  https://opencompass.openxlab.space/utils/VLMEval/MME.tsv
```

### Run

```bash
python3 scripts/mme_eval_local.py
```

Useful overrides:

```bash
python3 scripts/mme_eval_local.py \
  --tsv /tmp/MME.tsv \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B \
  --concurrency 8
```

### Result Interpretation

The bundled script outputs:

- `exact_acc`
- `unknown`
- official-style `MME` aggregate scores
- per-category scores

Pay special attention to:

- `unknown` count
- `numerical_calculation`
- `commonsense_reasoning`
- `OCR`

If `unknown` rises while raw model quality seems similar, first suspect prompt-following drift.

## L1 MMBench

### Dataset Objective

`MMBench_DEV_EN` is the second L1 benchmark in this skill because it provides broader MCQ coverage for:

- coarse and fine-grained perception
- attribute and relation reasoning
- logic reasoning
- OCR and structure reading
- localization and spatial reasoning
- future prediction

### Preparation

Download the official TSV if missing:

```bash
curl -k -L --fail --max-time 120 -o /tmp/MMBench_DEV_EN.tsv \
  https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv
```

### Run

```bash
python3 scripts/mmbench_eval_local.py
```

Useful overrides:

```bash
python3 scripts/mmbench_eval_local.py \
  --tsv /tmp/MMBench_DEV_EN.tsv \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B \
  --concurrency 8
```

### Result Interpretation

The bundled script outputs:

- `rows_all`
- `rows_scored`
- `exact_acc`
- `z_fallback`
- grouped accuracy summary close to `MMBench` heuristic evaluation

Pay special attention to:

- `LR`
- `future_prediction`
- `spatial_relationship`
- `object_localization`
- `structuralized_imagetext_understanding`
- `z_fallback`

If `z_fallback` rises, the model may still know the answer but fail to obey the "letter only" output constraint.

## Regression Policy

For routine tuning comparisons, treat the following as a healthy minimum process:

1. `L0` must pass fully.
2. `MME` should not show a clear drop in overall score or a spike in `unknown`.
3. `MMBench_DEV_EN` should not show a clear drop in overall score or a spike in `z_fallback`.
4. Investigate any category-specific regression even if the overall score looks flat.

Do not compare one run that used relaxed prompting against another run that used strong constrained prompting.

## Resources

- `scripts/l0_multimodal_smoke.py`
  Run the finalized image plus video L0 smoke suite.
- `scripts/multi_pics_eval.py`
  Run the deterministic `1` to `40` multi-image precision dataset with readiness gating and extraction-based scoring.
- `scripts/mme_eval_local.py`
  Run local `MME` with extraction-based scoring and official-style aggregation.
- `scripts/mmbench_eval_local.py`
  Run local `MMBench_DEV_EN` with letter extraction and circular-aware scoring.
- `scripts/run_full_regression.py`
  Run the standard `L0 -> MME -> MMBench_DEV_EN` regression chain and emit one final summary.
