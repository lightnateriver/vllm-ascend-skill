#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.5-4B model directory.}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-20}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$PROJECT_ROOT}"
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-0}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP="${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

async_flag=(--no-async-scheduling)
if [[ "$ENABLE_ASYNC_SCHEDULING" == "1" ]]; then
  async_flag=(--async-scheduling)
fi

exec vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --enforce-eager \
  --mm-processor-cache-type shm \
  --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --additional-config '{"enable_cpu_binding": true}' \
  "${async_flag[@]}"
