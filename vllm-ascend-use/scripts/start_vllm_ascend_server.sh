#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"

ASCEND_ENV_FILE="${ASCEND_ENV_FILE:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "${ASCEND_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ENV_FILE}"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
MM_PROCESSOR_CACHE_TYPE="${MM_PROCESSOR_CACHE_TYPE:-shm}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-20}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"
ENABLE_CPU_BINDING="${ENABLE_CPU_BINDING:-1}"

export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP="${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --enforce-eager
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ -n "${MM_PROCESSOR_CACHE_TYPE}" ]]; then
  cmd+=(--mm-processor-cache-type "${MM_PROCESSOR_CACHE_TYPE}")
fi

if [[ -n "${MM_PROCESSOR_CACHE_GB}" ]]; then
  cmd+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
fi

if [[ -n "${MM_ENCODER_TP_MODE}" ]]; then
  cmd+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}")
fi

if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH:-}" ]]; then
  cmd+=(--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}")
fi

if [[ "${ENABLE_CPU_BINDING}" == "1" ]]; then
  cmd+=(--additional-config '{"enable_cpu_binding": true}')
fi

if [[ "${ENABLE_ASYNC_SCHEDULING}" == "1" ]]; then
  cmd+=(--async-scheduling)
else
  cmd+=(--no-async-scheduling)
fi

printf 'Starting service command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}" "$@"
