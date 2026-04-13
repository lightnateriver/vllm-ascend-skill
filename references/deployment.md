# Deploy a Stock vLLM-Ascend Service

## Goal

Bring up a stock OpenAI-compatible service on Ascend NPU with a practical multimodal configuration:

- eager mode
- TP deployment
- multimodal SHM cache
- ViT data-parallel encoder mode
- CPU binding enabled
- HCCL environment tuned to a known-good baseline

## 1. Pre-flight checklist

Before starting the service, confirm all of these:

- `vllm` imports successfully.
- `vllm_ascend` imports successfully.
- The installed `vllm` and `vllm-ascend` versions are meant to be used together.
- Ascend runtime is available on the host.
- The target model exists locally.
- The directory that contains local images is known up front.

Useful quick checks:

```bash
python - <<'PY'
import vllm
import vllm_ascend
print("vllm ok")
print("vllm_ascend ok")
PY
```

## 2. Recommended baseline environment

These are pragmatic defaults for a controlled stock service run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

Treat these as deployment inputs. If you change them, note the change in benchmark or validation records.

## 3. Start the service with the bundled helper

The simplest way to start a stock service from this skill is:

```bash
export MODEL_PATH=/path/to/model
export ALLOWED_LOCAL_MEDIA_PATH=/path/to/local/media
export HOST=127.0.0.1
export PORT=8000
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=36864
export GPU_MEMORY_UTILIZATION=0.85

/root/.codex/skills/vllm-ascend-use/scripts/start_vllm_ascend_server.sh
```

The helper emits a plain `vllm serve` command with these default behaviors:

- `--enforce-eager`
- `--tensor-parallel-size 4` unless overridden
- `--mm-processor-cache-type shm`
- `--mm-processor-cache-gb 20`
- `--mm-encoder-tp-mode data`
- `--gpu-memory-utilization 0.85`
- `--max-model-len 36864`
- `--additional-config '{"enable_cpu_binding": true}'`

By default it also enables async scheduling for performance mode. Set:

```bash
export ENABLE_ASYNC_SCHEDULING=0
```

when you want the helper to emit `--no-async-scheduling` for a more controlled accuracy run.

## 4. Equivalent direct command

If you do not want to use the helper, the equivalent shape is:

```bash
vllm serve /path/to/model \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --mm-processor-cache-type shm \
  --mm-processor-cache-gb 20 \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 \
  --max-model-len 36864 \
  --allowed-local-media-path /path/to/local/media \
  --additional-config '{"enable_cpu_binding": true}' \
  --async-scheduling
```

## 5. What the important flags mean

- `--enforce-eager`
  Use eager mode as the baseline on Ascend for service bring-up, debugging, and controlled benchmarking.
- `--tensor-parallel-size`
  Split the model across NPUs.
- `--mm-processor-cache-type shm`
  Store multimodal preprocessing cache objects in shared memory.
- `--mm-processor-cache-gb`
  Reserve a capacity budget for the multimodal SHM cache.
- `--mm-encoder-tp-mode data`
  Use data-style distribution for the multimodal encoder across TP workers.
- `--allowed-local-media-path`
  Explicitly allow local file access for `file://` media references.
- `--additional-config '{"enable_cpu_binding": true}'`
  Enable CPU binding through vLLM additional config.
- `--async-scheduling`
  Performance-oriented scheduler mode. Keep it explicit so deployment intent is clear.
- `--no-async-scheduling`
  Preferred for a tighter accuracy comparison when your build supports Boolean optional flags.

## 6. Smoke test

After startup, confirm the service responds:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

If the service is multimodal and uses local files, also test a single image request before running a full benchmark.

## 7. Minimal multimodal request example

```json
{
  "model": "/path/to/model",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image briefly."},
        {"type": "image_url", "image_url": {"url": "file:///path/to/local/media/sample.png"}}
      ]
    }
  ],
  "temperature": 0,
  "max_completion_tokens": 64,
  "stream": true
}
```

## 8. Common deployment mistakes

- Forgetting to source the Ascend runtime environment.
- Installing mismatched `vllm` and `vllm-ascend` versions.
- Using `file://` images without `--allowed-local-media-path`.
- Omitting `--enforce-eager` while debugging or profiling.
- Changing performance-related environment variables between runs without recording the change.
