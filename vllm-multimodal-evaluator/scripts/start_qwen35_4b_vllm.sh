#!/usr/bin/env bash
# ============================================================================
# start_qwen35_4b_vllm.sh
#
# vLLM 通用启动脚本 — 适用于 NVIDIA GPU 和 Ascend NPU 环境。
# 默认开启：chunked prefill / async scheduling / prefix caching / function calling
# 默认启用：local media path / eager mode
#
# 用法:
#   MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_qwen35_4b_vllm.sh
#
# 环境变量:
#   MODEL_PATH                (必填) 模型本地路径
#   HOST                      (可选, 默认 127.0.0.1)
#   PORT                      (可选, 默认 8000)
#   GPU_MEMORY_UTILIZATION    (可选, 默认 0.7)
#   DTYPE                     (可选, 默认 bfloat16)
#   ALLOWED_LOCAL_MEDIA_PATH  (可选, 默认 $PROJECT_ROOT)
#   ENABLE_CHUNKED_PREFILL    (可选, 默认 1)
#   ENABLE_ASYNC_SCHEDULING   (可选, 默认 1)
#   ENABLE_PREFIX_CACHING     (可选, 默认 1)
#   ENABLE_FUNCTION_CALLING   (可选, 默认 1)
#   MAX_MODEL_LEN             (可选, 默认不限制)
#   EXTRA_ARGS                (可选, 额外 vllm 参数)
#
# 平台提示:
#   - NVIDIA GPU: source conda/venv，然后直接运行即可
#   - Ascend NPU: 运行前需 source ascnd-toolkit 环境变量，
#     并设置 ASCEND_RT_VISIBLE_DEVICES、VLLM_ASCEND_* 等
# ============================================================================
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local model directory.}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DTYPE="${DTYPE:-bfloat16}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$PROJECT_ROOT}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_FUNCTION_CALLING="${ENABLE_FUNCTION_CALLING:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# Build flag arrays
extra_flags=()

if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  extra_flags+=(--enable-chunked-prefill)
fi

if [[ "$ENABLE_ASYNC_SCHEDULING" == "1" ]]; then
  extra_flags+=(--async-scheduling)
else
  extra_flags+=(--no-async-scheduling)
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  extra_flags+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_FUNCTION_CALLING" == "1" ]]; then
  extra_flags+=(--enable-auto-tool-choice --tool-call-parser qwen3_xml)
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
  extra_flags+=(--max-model-len "$MAX_MODEL_LEN")
fi

# Determine entrypoint: prefer `vllm serve`, fallback to python -m
if command -v vllm &> /dev/null; then
  ENTRYPOINT=("vllm" "serve")
else
  echo "WARNING: 'vllm' not found in PATH. Using 'python -m vllm.entrypoints.openai.api_server'."
  ENTRYPOINT=("python" "-m" "vllm.entrypoints.openai.api_server")
fi

echo "=== Starting vLLM service ==="
echo "  Model:        $MODEL_PATH"
echo "  Host/Port:    $HOST:$PORT"
echo "  Dtype:        $DTYPE"
echo "  GPU Mem:      $GPU_MEMORY_UTILIZATION"
echo "  ChunkedPrefill:  $ENABLE_CHUNKED_PREFILL"
echo "  AsyncScheduling: $ENABLE_ASYNC_SCHEDULING"
echo "  PrefixCaching:   $ENABLE_PREFIX_CACHING"
echo "  FunctionCalling: $ENABLE_FUNCTION_CALLING"
echo "  LocalMediaPath:  $ALLOWED_LOCAL_MEDIA_PATH"
echo "  ExtraArgs:       ${EXTRA_ARGS:-}"
echo "==================================="

exec "${ENTRYPOINT[@]}" "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --dtype "$DTYPE" \
  --enforce-eager \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  "${extra_flags[@]}" \
  ${EXTRA_ARGS:-}
