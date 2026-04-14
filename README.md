# vllm-ascend-use

面向 Ascend NPU 场景的 `vLLM + vllm-ascend` 实战 skill，帮助使用者快速完成服务部署、性能测试，以及基于 `LLM` 前输入一致性的精度验证。

这个仓库适合下面几类需求：

- 需要理解 `vLLM V1` 与 `vllm-ascend` 的请求路径和整体结构
- 需要在 Ascend 环境中快速拉起一个 OpenAI 兼容的推理服务
- 需要生成并运行 `10k text token + 40 image` 的单并发多模态 benchmark
- 需要在关闭 `chunked prefill` 的前提下，对比两次部署在 `LLM` 前的输入是否一致

## 能力概览

本 skill 主要覆盖 4 个方向：

1. 架构理解
   说明从 API Server、Engine Core 到 TP Worker 的主要执行路径，以及 `vllm-ascend` 在 Ascend 平台上的适配位置。
2. 服务部署
   提供一套偏实战的 Ascend 部署基线，包括 eager mode、TP、SHM 多模态缓存、CPU binding 以及常见环境变量建议。
3. 性能测试
   支持生成 cache-safe 多模态测试集，并运行单并发 benchmark，输出 `TTFT`、`TPOT`、`E2E` 等指标。
4. 精度验证
   默认以“`LLM` 前输入是否一致”为主判据，而不是要求最终文本输出完全一致。

## 目录结构

```text
.
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── architecture.md
│   ├── deployment.md
│   ├── performance-testing.md
│   └── accuracy-testing.md
└── scripts/
    ├── start_vllm_ascend_server.sh
    ├── generate_multimodal_dataset.py
    ├── benchmark_chat_service.py
    ├── capture_chat_outputs.py
    └── compare_text_outputs.py
```

各目录职责如下：

- `SKILL.md`
  skill 的入口说明，定义适用场景、工作方式和关键约束。
- `references/`
  详细参考文档，分别覆盖架构、部署、性能测试和精度验证。
- `scripts/`
  用于减少重复操作的辅助脚本，便于直接拉服务、生成数据和驱动测试请求。
- `agents/`
  skill 相关的 agent 配置。

## 推荐使用方式

### 1. 构建上下文

根据任务类型优先阅读对应参考文档：

- 架构理解：`references/architecture.md`
- 服务部署：`references/deployment.md`
- 性能压测：`references/performance-testing.md`
- 精度验证：`references/accuracy-testing.md`

### 2. 优先使用 stock 路径

默认优先使用：

```bash
vllm serve
```

而不是：

```bash
python -m vllm.entrypoints.openai.api_server
```

这样更贴近 upstream `vllm` 与 `vllm-ascend` 的标准使用路径。

### 3. 精度测试的默认原则

这个 skill 当前推荐的精度验证方式是：

- 保持相同模型
- 保持相同输入集
- 保持相同请求顺序
- 串行发送请求
- 显式设置采样参数
- 在需要时关闭 `chunked prefill`
- 通过 runtime patch 或 monkeypatch 观测 `LLM` 前输入
- 以 `input_ids`、`inputs_embeds`、`positions` 等摘要或 hash 一致性作为主判据

默认不要求最终文本输出完全一致。

## 典型场景

### 场景 1：快速拉起 Ascend 推理服务

可直接复用：

```bash
scripts/start_vllm_ascend_server.sh
```

配合环境变量完成模型路径、端口、TP size、媒体目录等配置。

### 场景 2：运行单并发多模态 benchmark

先生成数据集，再执行 benchmark：

```bash
python scripts/generate_multimodal_dataset.py ...
python scripts/benchmark_chat_service.py ...
```

### 场景 3：验证两次部署在 LLM 前输入是否一致

建议使用：

- 固定数据集
- 固定请求顺序
- `temperature=0`
- `--no-async-scheduling`
- 关闭 `chunked prefill`
- 在 `NPUModelRunner._model_forward` 一类边界上插入运行时 patch

对齐比较的重点字段通常包括：

- `input_ids`
- `inputs_embeds`
- `positions`
- 少量稳定的 attention metadata

## 使用边界

这个 skill 的定位是“实战工作流”和“问题排查入口”，不是对所有 Ascend 组合配置做穷举说明。

使用时建议注意：

- `vllm` 与 `vllm-ascend` 版本要匹配
- 多模态本地文件访问需要正确设置 `--allowed-local-media-path`
- 进行输入一致性比较时，不要把时间戳、PID、进程 rank 等运行时噪声字段纳入比较 key
- 如果 baseline 自身已经不稳定，后续 candidate 的差异不能直接当成有效回归结论

## 版本说明

当前仓库内容面向 `vllm-ascend` `0.18` 分支维护，相关流程、参数和建议以该分支为准。

## 相关文件

- Skill 定义：`SKILL.md`
- 架构说明：`references/architecture.md`
- 部署说明：`references/deployment.md`
- 性能测试说明：`references/performance-testing.md`
- 精度测试说明：`references/accuracy-testing.md`

