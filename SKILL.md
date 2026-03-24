
### 1. 克隆 vllm-ascend

```bash
git clone https://github.com/vllm-project/vllm-ascend.git -b v0.13.0rc1 --depth 1
cd vllm-ascend
pip install -e . --no-deps
```

### 2. 应用补丁添加 qwen3_vl 支持

我们需要修改两个文件支持 Qwen3-VL 多模态模型：

**修改 1: `vllm_ascend/quantization/quant_config.py`**

需要修改：

1. **添加 qwen3_vl 前缀映射** (第 185 行附近)
```python
# key: model_type
# value: orig_to_new_prefix
QUANT_MODEL_PREFIX_MAPPINGS = {
    "qwen3_vl": {
        "visual.": "visual.",
        "language_model.lm_head.": "lm_head.",
        "language_model.model.": "model.",
    },
    "qwen3_vl_moe": {
        "visual.": "model.visual.",
        "language_model.lm_head.": "lm_head.",
        "language_model.model.": "model.language_model.",
    },
}
```

2. **添加 qwen3_vl 融合模块映射** (第 200 行附近)
```python
packed_modules_model_mapping = {
    "qwen3_vl": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    },
    "qwen3_moe": {
        # ... existing content ...
```

3. **修复 `is_layer_skipped_ascend`** 处理缺失层：

```python
    def is_layer_skipped_ascend(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]] = MappingProxyType({})):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                key = shard_prefix + '.weight'
                if key not in self.quant_description:
                    # If any shard is not found, entire fused layer is not quantized
                    # Mark as not skipped (not quantized)
                    return False
                is_shard_skipped = self.quant_description[key] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            key = prefix + '.weight'
            if key not in self.quant_description:
                # Layer not found means this layer is not quantized
                # return False = not skipped, use unquantized
                return False
            is_skipped = self.quant_description[key] == "FLOAT"

        assert is_skipped is not None
        return is_skipped
```

4. **修复 `create_weights` 和 `apply`** 处理 `quant_method is None`：

在 `AscendLinearMethod.create_weights` 开始添加：
```python
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if self.quant_method is None:
            # This layer is not quantized, create standard weight
            weight = ModelWeightParameter(
                torch.empty(
                    (sum(output_partition_sizes), input_size_per_partition),
                    dtype=params_dtype,
                ),
                requires_grad=False,
                output_dim=0,
                input_dim=1,
            )
            set_weight_attrs(weight, extra_weight_attrs)
            layer.register_parameter("weight", weight)
            return
        # ... existing code continues below ...
```

在 `AscendLinearMethod.apply` 开始添加：
```python
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.quant_method is None:
            # This layer is not quantized, directly compute with existing weight
            if hasattr(layer, 'weight'):
                weight = layer.weight
            elif hasattr(layer, '_weight'):
                weight = layer._weight
            else:
                raise RuntimeError(f"Could not find weight on layer {layer}")
            if bias is not None:
                return x @ weight.T + bias
            return x @ weight.T
        # ... existing code continues below ...
```

**修改 2: `vllm_ascend/quantization/utils.py`**

修复 `get_linear_quant_type` 处理缺失键：

```python
def get_linear_quant_type(quant_description: Dict[str, Any], prefix: str,
                          packed_modules_mapping: Dict[str, Any]):
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            key = shard_prefix + '.weight'
            if key not in quant_description:
                # If any shard not found, entire layer is not quantized
                return None
            shard_quant_type = quant_description[key]

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                raise ValueError(
                    f"Not all shards of {prefix} are quantized with same quant type."
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard "
                    f"use {quant_type}. Please check quantization config.")
    else:
        key = prefix + '.weight'
        if key not in quant_description:
            # Layer not found means this layer is not quantized
            return None
        quant_type = quant_description[key]
    return quant_type
```

修复 `get_quant_method_modelslim` 处理 `quant_type is None`：

```python
def get_quant_method_modelslim(
        quant_description: Dict[str, Any],
        prefix: str,
        layer_type: str,
        packed_modules_mapping: Optional[Dict[str, Any]] = None):
    logger.info_once("Using the vLLM Ascend modelslim Quantization now!")
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()
    # Attention
    if '.attn' in prefix and 'fa_quant_type' in quant_description.keys():
        quant_type = quant_description['fa_quant_type']
    # Linear
    else:
        quant_type = get_linear_quant_type(quant_description, prefix,
                                           packed_modules_mapping)
    if quant_type is None:
        # This layer is not quantized, return None
        return None
    if quant_type in ASCEND_QUANTIZATION_METHOD_MAP.keys():
```

### 3. 修复依赖导入

确保在 `vllm_ascend/quantization/quant_config.py` 顶部正确导入：

```python
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               RowParallelLinear,
                                               ModelWeightParameter,
                                               set_weight_attrs)
```

### 4. 创建启动脚本

```python
#!/usr/bin/env python
"""
Start bf16 non-quantized Qwen3-VL-8B-Instruct on Ascend NPU
"""

import os
import subprocess

os.environ['VLLM_USE_MODELSCOPE'] = 'True'
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:256'

cmd = [
    'vllm', 'serve',
    '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
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

保存为 `start_qwen3vl_bf16_optimized.py` 然后启动：

```bash
python start_qwen3vl_bf16_optimized.py
```

## 遇到的问题和解决方案

### 问题 1: KeyError: key not found in quantization_description

**原因**: Qwen3-VL 的 vision tower 层不在量化描述中，原始代码遇到缺失键直接抛出错误。

**解决**: 添加错误处理，如果键不在量化描述中，认为该层不需要量化，正确加载未量化权重。

### 问题 2: 混合缩进错误

**原因**: 增量修改时，原文件用 4 空格缩进，增量修改偶尔会导致缩进不对。

**解决**: 需要逐段检查对齐，确保所有缩进都是 4 空格。

### 问题 3: NPU OOM 显存不足

**原因**: 8B BF16 需要较大显存，设置较高 `gpu-memory-utilization` 会导致分配失败。

**解决**: 调整到 `gpu-memory-utilization=0.75`，给 vLLM 内核和缓存留出空间，成功启动。

### 问题 4: AttributeError: 'AscendQKVParallelLinear' object has no attribute 'weight'

**原因**: 当 `quant_method is None` (不需要量化)，原始代码没有创建权重属性。

**解决**: 添加 `create_weights` 创建标准权重，注册到 layer 解决。

### 问题 5: AttributeError: 'NoneType' object has no attribute 'apply'

**原因**: 当 `quant_method is None`，原始代码仍然调用 `self.quant_method.apply()`。

**解决**: 在 `apply` 方法开始添加检查，如果 `self.quant_method is None`，直接用权重矩阵计算输出。

## API 使用

启动成功后，可以通过 OpenAI 兼容 API 调用：

```python
import openai
client = openai.OpenAI(
    base_url="http://localhost:13700/v1",
    api_key="token-xxx",
)

# Text chat
response = client.chat.completions.create(
    model="qwen3-vl-8b-Instruct",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    max_tokens=128,
)
print(response)
```

## 修改文件总结

- ✅ `vllm_ascend/quantization/quant_config.py` - 添加 qwen3_vl 支持 + 缺失错误处理
- ✅ `vllm_ascend/quantization/utils.py` - 添加缺失键处理
- ✅ `deploy_qwen3vl_bf16.md` - 完整部署总结

## 验收

- [x] Qwen3-VL-8B-Instruct BF16 可以正确启动
- [x] 能够正确处理 vision tower 不量化情况
- [x] API 服务正常监听，可以正常推理