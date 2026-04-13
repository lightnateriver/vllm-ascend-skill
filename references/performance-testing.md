# Performance Testing Playbook

## Target case

Use this playbook for the standard single-concurrency multimodal case:

- `10,000` text tokens
- `40` images per request
- image size `288x512`
- `13` request groups total
- `3` warmup groups
- `10` measured groups

The core requirement is to avoid accidental cache hits by making all text and image content unique:

- no repeated text lines inside a request
- no repeated text bodies across requests
- no repeated images inside a request
- no repeated image bytes across requests

## 1. Generate the dataset

Use the bundled generator:

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/generate_multimodal_dataset.py \
  --tokenizer /path/to/model \
  --request-model /path/to/model \
  --output-dir /tmp/vllm_ascend_mm_bench \
  --rounds 13 \
  --warmup-rounds 3 \
  --images-per-round 40 \
  --width 288 \
  --height 512 \
  --target-text-tokens 10000
```

What it creates:

- `round_0` to `round_12`
- `payload.json` for each round
- `images/` directory for each round
- `dataset_manifest.json` with uniqueness metadata

Check the manifest after generation:

- `unique_text_count` should equal `13`
- `unique_image_count` should equal `13 x 40 = 520`

## 2. Start the service in performance mode

Start the service with the deployment helper and keep async scheduling enabled unless you are doing a stability study:

```bash
export MODEL_PATH=/path/to/model
export ALLOWED_LOCAL_MEDIA_PATH=/tmp/vllm_ascend_mm_bench
export TENSOR_PARALLEL_SIZE=4
export ENABLE_ASYNC_SCHEDULING=1

/root/.codex/skills/vllm-ascend-use/scripts/start_vllm_ascend_server.sh
```

## 3. Run the benchmark

Use the bundled benchmark script:

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/benchmark_chat_service.py \
  --host http://127.0.0.1:8000 \
  --data-dir /tmp/vllm_ascend_mm_bench \
  --tokenizer /path/to/model \
  --label baseline \
  --out /tmp/vllm_ascend_mm_bench_perf.json
```

The benchmark is serial and single-concurrency by design. It reports:

- `TTFT`
- `TPOT`
- `E2E`

for both:

- `mean`
- `p90`
- `p95`
- `p99`

## 4. Keep output length under control when needed

If you need cleaner TPOT or E2E comparison across multiple runs, clamp the output length explicitly:

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/benchmark_chat_service.py \
  --host http://127.0.0.1:8000 \
  --data-dir /tmp/vllm_ascend_mm_bench \
  --tokenizer /path/to/model \
  --label baseline_cap72 \
  --max-completion-tokens 72 \
  --out /tmp/vllm_ascend_mm_bench_perf_cap72.json
```

This does not make the model deterministic, but it removes one common source of metric drift: different completion lengths.

## 5. Metric definitions used by the benchmark script

- `TTFT`
  Time from request send to first streamed text token.
- `TPOT`
  `(last_text_token_time - first_text_token_time) / max(text_output_tokens - 1, 1)`
- `E2E`
  Time from request send to request end.

The script computes token counts from the captured output text with the provided tokenizer, so metric math does not depend on the server's `usage` field being perfect.

## 6. Benchmark interpretation rules

- Use only measured rounds for summary statistics. Warmup rounds are not part of the final aggregate.
- Keep the same dataset, deployment command, and environment variables across comparison runs.
- If you change TP size, model length, async scheduling, or output cap, treat the result as a different benchmark configuration.
- If exact latency numbers matter, record the full start command and the exported environment variables beside the result JSON.

## 7. Recommended artifacts to save

Save at least these:

- benchmark result JSON
- dataset manifest
- exact service start command
- shell exports used for the run
- `curl /v1/models` response for the running service

That is enough to reproduce most baseline runs.
