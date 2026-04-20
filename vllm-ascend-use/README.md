# vllm-ascend-use

面向 Ascend NPU 场景的 `vLLM + vllm-ascend` 通用实战 skill，帮助使用者快速完成服务部署、性能测试，以及基于 `LLM` 前输入一致性的精度验证。

## 适用场景

这个 skill 适合下面几类任务：

- 理解 `vLLM V1` 与 `vllm-ascend` 的请求路径和整体结构
- 在 Ascend 环境中拉起一个 OpenAI 兼容的推理服务
- 生成并运行 `10k text token + 40 image` 的单并发多模态 benchmark
- 对比两次部署在 `LLM` 前的输入是否一致

如果任务明确是 API server 热点分析，请改用同仓库里的 `vllm-ascend-api-server-profiler`。
如果任务明确是构造简单多模态测试数据并跑能力支持矩阵，请改用同仓库里的 `vllm-multimodal-evaluator`。可以把它视作本 skill 的配套子 skill。

## 能力概览

本 skill 主要覆盖四个方向：

1. 架构理解
   说明从 API server、engine core 到 TP worker 的主要执行路径，以及 `vllm-ascend` 在 Ascend 平台上的适配位置。
2. 服务部署
   提供偏实战的 Ascend 部署基线，包括 eager mode、TP、SHM 多模态缓存、CPU binding 以及常见环境变量建议。
3. 性能测试
   支持生成 cache-safe 多模态测试集，并运行单并发 benchmark，输出 `TTFT`、`TPOT`、`E2E` 等指标。
4. 精度验证
   默认以“`LLM` 前输入是否一致”为主判据，而不是要求最终文本输出完全一致。
5. 多模态能力评估联动
   当目标从“服务是否能跑起来”切换成“服务支持哪些图片和视频输入方式、格式和时序理解能力”时，交给 `vllm-multimodal-evaluator`。

## 目录结构

```text
.
├── README.md
├── SKILL.md
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

## 推荐使用方式

### 1. 先构建上下文

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

### 3. 精度验证的默认原则

推荐保持：

- 相同模型
- 相同输入集
- 相同请求顺序
- 串行请求
- 显式采样参数
- 在需要时关闭 `chunked prefill`
- 以 `LLM` 前输入的一致性作为主判据

## 相关文件

- Skill 定义：`SKILL.md`
- 架构说明：`references/architecture.md`
- 部署说明：`references/deployment.md`
- 性能测试说明：`references/performance-testing.md`
- 精度测试说明：`references/accuracy-testing.md`
- 配套子 skill：`../vllm-multimodal-evaluator/`
