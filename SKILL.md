---
name: vllm-ascend-opt
description: One-click deployment and inference optimization for vllm-ascend supported models on Ascend NPU. Provides optimized configuration suggestions and automatic patch for common issues. Use when deploying or optimizing vLLM inference on Ascend hardware with popular models like Qwen3-VL, Qwen3, DeepSeek-V3, etc.
---

# vllm-ascend-opt

One-click deployment and inference optimization for vllm-ascend supported models on Ascend NPU.

## Overview

This skill automates the deployment and inference optimization workflow for models running on vllm-ascend (vLLM Ascend Plugin). It includes:

1. Automatic patching for common issues (like partial quantization in multimodal models)
2. Suggests optimized configuration parameters based on model size and hardware
3. Generates ready-to-use startup scripts for different model types
4. Provides debugging guidance for common deployment problems

## Core Capabilities

### 1. One-Click Model Deployment

Automatically generate and configure startup scripts based on model type:
- Qwen3-VL multimodal models
- Qwen3 dense / MoE models
- DeepSeek-V3 / DeepSeek-R1 models
- Qwen2.5-VL / Omni models
- GLM-4.5 models
- Kimi-K2-Thinking models

### 2. Inference Optimization

Provides optimized configuration recommendations for:
- Memory allocation (`PYTORCH_NPU_ALLOC_CONF`)
- GPU memory utilization (`gpu-memory-utilization`)
- CUDAGraph mode (`cudagraph_mode`)
- Max sequence length and batch sizing
- Quantization strategies (W8A8, W4A8, W4A4, etc.)
- Expert parallelism and data parallelism configurations

### 3. Troubleshooting Common Issues

Patches and solutions for known issues:
- Partial quantization handling (only LLM backbone quantized, vision tower not quantized)
- Missing key errors in quantization config
- Memory fragmentation on NPU
- ACL graph compilation issues

## Workflow

### Step 1: Model Analysis
When given a model path and type:
1. Check model architecture (dense/MoE/multimodal)
2. Identify quantization configuration
3. Determine hardware capabilities (NPU memory available)

### Step 2: Configuration Generation
Generate optimized startup script based on:
- Model size
- Quantization method
- Number of available NPUs
- Target usage (online serving/offline inference)

Example output for Qwen3-VL-8B-Instruct BF16:
```python
#!/usr/bin/env python
"""
Start bf16 non-quantized version of Qwen3-VL-8B-Instruct with optimized settings
"""

import os
import subprocess

os.environ['VLLM_USE_MODELSCOPE'] = 'True'
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:256'

cmd = [
    'vllm', 'serve',
    '/path/to/Qwen3-VL-8B-Instruct',
    '--host', '0.0.0.0',
    '--port', '13700',
    '--data-parallel-size', '1',
    '--tensor-parallel-size', '1',
    '--seed', '1024',
    '--served-model-name', 'qwen3-vl-8b-Instruct',
    '--max-model-len', '6144',
    '--max-num-batched-tokens', '6144',
    '--max-num-seqs', '1',
    '--trust-remote-code',
    '--async-scheduling',
    '--enforce-eager',
    '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    '--mm_processor_cache_type', 'shm',
    '--gpu-memory-utilization', '0.75',
    '--no-enable-prefix-caching',
]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)
```

### Step 3: Automatic Patching

Apply necessary patches to vllm-ascend source code for known issues:

| Issue | Patch Location | Description |
|-------|----------------|-------------|
| Partial quantization (Qwen3-VL) | `vllm_ascend/quantization/quant_config.py` | Add missing error handling for unquantized layers |
| Partial quantization (Qwen3-VL) | `vllm_ascend/quantization/utils.py` | Fix null quantization type handling |

### Step 4: Verification

After deployment, verify:
- Model loads successfully
- API endpoint responds
- Inference works correctly

## Optimization Guidelines

### Memory Allocation

| NPU Memory | Recommended `max_split_size_mb` | Notes |
|------------|---------------------------------|-------|
| < 64GB | 256 | Reduces fragmentation for small allocations |
| ≥ 64GB | 512 | Balances fragmentation and allocation speed |

### GPU Memory Utilization

| Model Type | Recommended `gpu-memory-utilization` |
|------------|--------------------------------------|
| BF16 (non-quantized) | 0.75 - 0.8 | Leave headroom for intermediate tensors |
| Quantized (W8A8/W4A16) | 0.85 - 0.9 | More memory available for KV cache |

### CUDAGraph Settings

| Scenario | Recommended `cudagraph_mode` |
|----------|-------------------------------|
| Online serving | `FULL_DECODE_ONLY` | Best latency for decode phase |
| Offline batch inference | `OFF` | Better throughput for variable batch sizes |

## Usage Examples

**Example 1: User asks to optimize Qwen3-VL model**
> "帮我优化vllm-ascend的qwen3vl模型"

Process:
1. Identify model: Qwen3-VL-8B-Instruct multimodal, BF16, only LLM backbone needs quantization
2. Apply patches to `quant_config.py` and `utils.py` for partial quantization support
3. Generate optimized startup script with appropriate memory settings
4. Provide instructions for starting service and API testing

**Example 2: User asks to deploy Qwen3-30B with W8A8 quantization**

Process:
1. Check hardware: 2 NPUs needed for 30B with W8A8
2. Recommend tensor parallel size = 2
3. Generate optimized config with proper memory settings
4. Provide startup command

## Resources

### references/
- `deployment-examples/`: Reference deployment configurations for popular models

### scripts/
- Utility scripts for common optimization tasks

