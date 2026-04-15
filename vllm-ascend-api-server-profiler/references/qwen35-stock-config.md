# Stock Qwen3.5 API Server Profiling Baseline

## Environment

Export these variables for the stock Qwen3.5 multimodal profile:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_NZ=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## Serve arguments

Use these stock serving arguments unless the user asks for a different baseline:

```bash
--tensor-parallel-size 4
--gpu-memory-utilization 0.85
--max-model-len 36864
--max-num-batched-tokens 36864
--async-scheduling
--additional-config '{"enable_cpu_binding": true}'
--mm-processor-cache-type shm
--mm-processor-cache-gb 20
--mm-encoder-tp-mode data
```

Also set:

```bash
--allowed-local-media-path <dataset_root>
```

## Dataset shape

- `3 warmup + 1 profile`
- Single request per round
- `10k` text tokens plus `40` images
- Image size `288x512`
- No repeated text or image content within or across the four rounds

## Required profiling constraints

- Do not modify `vllm` source.
- Do not modify `vllm-ascend` source.
- Do not use `0407` patch content.
- Do not use phase-based profiling schemes.
- Install all timing logic through external monkey patch only.

## Suggested target functions

Track these API server functions before widening the scope:

- `HfRenderer.render_messages_async`
- `parse_chat_messages_async`
- `MediaConnector.load_from_url_async`
- `AsyncMicrobatchTokenizer.encode`
- `ProcessorInputs.get_mm_hashes`
- `InputProcessingContext.call_hf_processor`
- `Qwen3VLMultiModalProcessor._call_hf_processor`
- `Qwen3VLProcessor.__call__`
- `Qwen2VLImageProcessorFast.__call__`
- `Qwen2VLImageProcessorFast.preprocess`
- `Qwen2VLImageProcessorFast._preprocess`
- `Qwen2TokenizerFast.__call__`
- `BatchFeature.convert_to_tensors`
- `BatchEncoding.convert_to_tensors`
- `SingleWriterShmObjectStorage.copy_to_buffer`

## Outer API server span

Use this request-scoped span as the API server preprocessing total:

- Start: `HfRenderer.render_messages_async`
- End: `SingleWriterShmObjectStorage.copy_to_buffer`
- Label: `api_server_total.render_messages_async_to_copy_to_buffer`
