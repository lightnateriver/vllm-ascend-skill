# Accuracy and Output-Consistency Testing

## Goal

Use an unmodified stock service as the baseline, capture its outputs in fixed request order, then compare later runs against that baseline.

The emphasis is consistency, not only correctness:

- same dataset
- same request order
- same model
- same main serving flags
- same sampling parameters
- no concurrent requests

## 1. Use the same dataset for both sides

Generate one dataset and reuse it for both baseline and candidate runs:

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/generate_multimodal_dataset.py \
  --tokenizer /path/to/model \
  --request-model /path/to/model \
  --output-dir /tmp/vllm_ascend_mm_bench
```

Do not regenerate the dataset between baseline and candidate unless you also accept that the comparison changed.

## 2. Prefer a controlled service configuration

For accuracy comparison, use the most stable configuration you can get:

- `temperature=0`
- explicit `max_completion_tokens`
- serial requests only
- fixed request order
- prefer `--no-async-scheduling` if your build supports it
- keep the same TP size between baseline and candidate

The bundled start helper supports this:

```bash
export MODEL_PATH=/path/to/model
export ALLOWED_LOCAL_MEDIA_PATH=/tmp/vllm_ascend_mm_bench
export ENABLE_ASYNC_SCHEDULING=0

/root/.codex/skills/vllm-ascend-use/scripts/start_vllm_ascend_server.sh
```

## 3. Capture baseline outputs

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/capture_chat_outputs.py \
  --host http://127.0.0.1:8000 \
  --data-dir /tmp/vllm_ascend_mm_bench \
  --label baseline \
  --max-completion-tokens 72 \
  --seed 0 \
  --out /tmp/baseline_outputs.json
```

This script:

- walks `round_0` to `round_12` in order
- sends one request at a time
- captures streamed text
- stores the final server response

## 4. Capture candidate outputs

After switching to the candidate service, run the same command with a different label and output path:

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/capture_chat_outputs.py \
  --host http://127.0.0.1:8000 \
  --data-dir /tmp/vllm_ascend_mm_bench \
  --label candidate \
  --max-completion-tokens 72 \
  --seed 0 \
  --out /tmp/candidate_outputs.json
```

## 5. Compare by round

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/compare_text_outputs.py \
  --left /tmp/baseline_outputs.json \
  --right /tmp/candidate_outputs.json \
  --out /tmp/baseline_vs_candidate.json
```

The compare script flags per-round text mismatches and shows preview snippets.

## 6. Baseline self-check comes first

Before concluding that the candidate regressed, verify the baseline against itself:

1. Start the stock service.
2. Capture `baseline_run1.json`.
3. Capture `baseline_run2.json`.
4. Compare them.

If baseline vs baseline already mismatches, exact-text comparison is not a clean regression signal yet.

In that situation:

- keep the requests serial
- keep `temperature=0`
- clamp `max_completion_tokens`
- prefer `--no-async-scheduling`
- consider reducing distributed complexity for diagnosis

## 7. Practical expectations

Even when `temperature=0`, a distributed multimodal serving stack can still show output drift if the runtime path is not fully deterministic. That is why the baseline self-check is a required step in serious validation work.

Use this interpretation order:

1. Baseline vs baseline stable
   Candidate mismatches are meaningful.
2. Baseline vs baseline unstable
   Fix or reduce nondeterminism first, then compare again.

## 8. Minimum artifacts to keep

- dataset manifest
- baseline capture JSON
- candidate capture JSON
- compare JSON
- exact start commands for both runs
- key environment variables for both runs

Without those artifacts, later accuracy disputes are hard to resolve.
