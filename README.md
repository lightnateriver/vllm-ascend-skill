# vllm-ascend-opt

面向 `vllm + vllm-ascend` 的多 skill 仓库，当前聚焦 Ascend NPU 场景下的部署、测试与 API server 侧性能分析。

这个仓库当前包含两个并列 skill：

- `vllm-ascend-use`
  面向 stock `vllm-ascend` 的通用实战工作流，覆盖架构理解、服务部署、性能测试和前 `LLM` 输入一致性验证。
- `vllm-ascend-api-server-profiler`
  面向 stock `vllm-ascend` OpenAI API server 的热点分析工作流，强调外置 monkey patch、请求级 profile、stage-based breakdown，以及基于热点函数内部逻辑的多方案调优分析。

## 仓库结构

```text
.
├── README.md
├── vllm-ascend-use/
│   ├── README.md
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
└── vllm-ascend-api-server-profiler/
    ├── README.md
    ├── SKILL.md
    ├── agents/
    ├── references/
    └── scripts/
```

## 适用场景

适合下面几类需求：

- 需要解释 `vLLM V1`、`vllm-ascend` 与 Ascend NPU 的执行路径
- 需要快速部署一个 stock `vllm serve` 服务并跑通多模态请求
- 需要生成 `10k text token + 40 image` 的单并发 benchmark 数据并做性能测试
- 需要比较两次部署在 `LLM` 前的输入是否一致
- 需要分析 OpenAI API server 侧而不是 engine core / TP worker 侧的热点函数
- 需要先拿到 API server 热点函数的多种调优方案，再决定后续实际优化路线

## 使用建议

### 1. 先选 skill

- 如果任务是通用部署、benchmark 或输入一致性验证，优先使用 `vllm-ascend-use`
- 如果任务是 API server 热点定位、timeline 生成或热点函数解释，优先使用 `vllm-ascend-api-server-profiler`
- 如果任务已经进入“为什么合计时间对不上”的阶段，优先使用 `vllm-ascend-api-server-profiler`，因为它会先拆 `stage_spans`、再解释 `concurrency_windows` 和 stage gap

### 2. 从同一个 GitHub 仓库安装多个 skill

如果要从这个仓库安装两个 skill，可以使用：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-ascend-use vllm-ascend-api-server-profiler \
  --ref 0.18
```

如果只安装其中一个，也可以只传单个 `--path`。

## 版本说明

当前仓库内容面向 `0.18` 分支维护。与 `vllm`、`vllm-ascend`、Ascend 环境相关的默认参数和工作流，以该分支中的 skill 内容为准。
