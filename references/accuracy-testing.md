# Accuracy and Pre-LLM Input-Consistency Testing

## Goal

Use an unmodified stock service as the baseline, drive both services with the same requests in fixed order, then compare whether the inputs immediately before the LLM are consistent between runs.

The emphasis is consistency at the model-input boundary:

- same dataset
- same request order
- same model
- same main serving flags
- same sampling parameters
- no concurrent requests
- same pre-LLM tensors or equivalent summarized traces

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
- disable `chunked prefill` when the purpose is to compare pre-LLM inputs

The bundled start helper supports this:

```bash
export MODEL_PATH=/path/to/model
export ALLOWED_LOCAL_MEDIA_PATH=/tmp/vllm_ascend_mm_bench
export ENABLE_ASYNC_SCHEDULING=0

/root/.codex/skills/vllm-ascend-use/scripts/start_vllm_ascend_server.sh
```

When comparing pre-LLM inputs, add a runtime patch or monkeypatch layer instead of editing `vllm` or `vllm-ascend` source directly. A practical hook point is `vllm_ascend.worker.model_runner_v1.NPUModelRunner._model_forward`, where you can summarize and hash inputs such as:

- `input_ids` when present
- `inputs_embeds` when the model path feeds embeddings directly
- `positions`
- selected attention metadata fields when they are stable and relevant

Record only the fields that represent the boundary immediately before the LLM. Avoid including timestamps, PIDs, or other run-specific metadata in the comparison key.

## 3. Drive baseline requests and capture pre-LLM traces

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/capture_chat_outputs.py \
  --host http://127.0.0.1:8000 \
  --data-dir /tmp/vllm_ascend_mm_bench \
  --label baseline \
  --max-completion-tokens 72 \
  --seed 0 \
  --out /tmp/baseline_driver_outputs.json
```

This script:

- walks `round_0` to `round_12` in order
- sends one request at a time
- is a convenient fixed-order request driver for the traced service
- may store final server responses, but those responses are not the primary comparison target for this workflow

At the same time, your runtime patch should append summarized pre-LLM records to a trace file such as:

```text
/tmp/baseline_pre_llm_trace.jsonl
```

Each line should contain only comparison-relevant fields such as round or call index, token count, and hashes of the pre-LLM tensors.

## 4. Drive candidate requests and capture pre-LLM traces

After switching to the candidate service, run the same command with a different label and output path:

```bash
python /root/.codex/skills/vllm-ascend-use/scripts/capture_chat_outputs.py \
  --host http://127.0.0.1:8000 \
  --data-dir /tmp/vllm_ascend_mm_bench \
  --label candidate \
  --max-completion-tokens 72 \
  --seed 0 \
  --out /tmp/candidate_driver_outputs.json
```

and capture the candidate trace, for example:

```text
/tmp/candidate_pre_llm_trace.jsonl
```

## 5. Compare the pre-LLM traces

Compare by round or call index, using only the stable pre-LLM fields:

- `input_ids` hash when present
- `inputs_embeds` hash when present
- `positions` hash
- optionally a minimal set of stable attention metadata hashes

If those hashes match for the aligned requests, treat the pre-LLM inputs as consistent.

Exact text-output comparison can still be run as a secondary diagnostic, but it is optional and should not be the default pass or fail criterion in this workflow.

## 6. Baseline self-check comes first

Before concluding that the candidate regressed, verify the baseline against itself:

1. Start the stock service.
2. Drive `baseline_run1` with the same fixed request sequence and capture `baseline_run1_pre_llm.jsonl`.
3. Drive `baseline_run2` with the same fixed request sequence and capture `baseline_run2_pre_llm.jsonl`.
4. Compare those two pre-LLM trace files.

If baseline vs baseline already mismatches at the pre-LLM input level, later candidate mismatches are not a clean regression signal yet.

In that situation:

- keep the requests serial
- keep `temperature=0`
- clamp `max_completion_tokens`
- prefer `--no-async-scheduling`
- disable `chunked prefill`
- consider reducing distributed complexity for diagnosis

## 7. Practical expectations

Even when `temperature=0`, a distributed multimodal serving stack can still show output drift after the LLM boundary. For this skill, that downstream drift is not enough by itself to prove an accuracy regression if the pre-LLM inputs are already consistent.

Use this interpretation order:

1. Baseline vs baseline stable
   Candidate pre-LLM mismatches are meaningful.
2. Baseline vs baseline unstable
   Fix or reduce nondeterminism first, then compare again.

## 8. Minimum artifacts to keep

- dataset manifest
- baseline pre-LLM trace JSONL
- candidate pre-LLM trace JSONL
- pre-LLM compare JSON
- exact start commands for both runs
- key environment variables for both runs

Without those artifacts, later accuracy disputes are hard to resolve.
