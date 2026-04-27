# vllm-ascend-opt

面向 `vllm + vllm-ascend` 的多 skill 仓库，当前聚焦 Ascend NPU 场景下的部署、测试与 API server 侧性能分析。

这个仓库当前包含四个并列 skill：

- `vllm-ascend-use`
  面向 stock `vllm-ascend` 的通用实战工作流，覆盖架构理解、服务部署、性能测试和前 `LLM` 输入一致性验证。
- `vllm-ascend-api-server-profiler`
  面向 stock `vllm-ascend` OpenAI API server 的热点分析工作流，强调外置 monkey patch、请求级 profile、stage-based breakdown，以及基于热点函数内部逻辑的多方案调优分析。
- `vllm-multimodal-evaluator`
  面向 stock `vllm` 或 `vllm-ascend` OpenAI 兼容服务的多模态能力评估工作流，覆盖本地图片和视频测试数据生成、Qwen3.5-4B 本地媒体部署，以及图片格式、Base64、多图、图文穿插、视频格式和视频时序理解 checklist。
- `vllm-multimodal-precision-testing`
  面向本地 `vLLM` 或 `vllm-ascend` OpenAI 兼容服务的多模态精度回归工作流，覆盖 `L0` 固定图片和视频冒烟测试、`L0.5` 的 `1~40` 张图多图精度测试，以及 `L1` 的 `MME` 与 `MMBench_DEV_EN` 回归测试，适合在模型调优或部署变更后做快速、可重复的多模态精度检查。

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
├── vllm-multimodal-evaluator/
│   ├── SKILL.md
│   ├── README.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── vllm-multimodal-precision-testing/
│   ├── README.md
│   ├── SKILL.md
│   ├── agents/
│   ├── assets/
│   ├── multi-pics-datasets/
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
- 需要生成规则化图像或视频测试数据，并对多模态服务做能力支持矩阵验证
- 需要在模型调优后执行固定冒烟用例、多图 `1~40` 回归，以及 `MME` / `MMBench` 回归测试
- 需要保留逐题问题、标准答案、模型回答和抽取结果，便于后续误差分析

## 使用建议

### 1. 先选 skill

- 如果任务是通用部署、benchmark 或输入一致性验证，优先使用 `vllm-ascend-use`
- 如果任务是 API server 热点定位、timeline 生成或热点函数解释，优先使用 `vllm-ascend-api-server-profiler`
- 如果任务是多模态能力评估、图片或视频支持矩阵验证，优先使用 `vllm-multimodal-evaluator`
- 如果任务是多模态精度回归、固定冒烟测试或 `MME` / `MMBench` 回归，优先使用 `vllm-multimodal-precision-testing`
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

如果要连同多模态能力评估 skill 一起安装，可以使用：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-ascend-use vllm-multimodal-evaluator vllm-ascend-api-server-profiler \
  --ref 0.18
```

如果要把多模态精度回归 skill 一起装上，推荐使用：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-ascend-use vllm-multimodal-evaluator vllm-multimodal-precision-testing vllm-ascend-api-server-profiler \
  --ref 0.18
```

如果只安装多模态精度回归 skill，也可以使用：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-multimodal-precision-testing \
  --ref 0.18
```

## 新增 skill 说明

### vllm-multimodal-precision-testing

这是仓库中的标准子 skill，目标是为多模态模型提供一个“先冒烟、再基准”的分层回归方案。

它的核心用途包括：

- 用固定 `L0` 图片和视频 case 快速发现明显退化
- 用内置 `1~40` 张图数据集观察单请求多图定位与检索能力
- 用 `MME` 做 yes/no 感知与推理回归
- 用 `MMBench_DEV_EN` 做 MCQ 感知与推理回归
- 保存逐题结果，便于分析某一次调优到底影响了哪些题型

这个子 skill 现在还直接自带：

- `L0` 冒烟测试图片和视频资源
- `L0.5` 多图精度测试数据集
- 多图测试脚本 `scripts/multi_pics_eval.py`

其中多图测试脚本默认支持 readiness gating，要求：

1. `/v1/models` 返回 `200`
2. 一个最小 text chat 请求也返回 `200`

这样可以避免服务启动早期的 `502` 噪声污染精度结果。

推荐把它和 `vllm-multimodal-evaluator` 配套使用：

- `vllm-multimodal-evaluator` 负责生成测试数据和做能力支持矩阵验证
- `vllm-multimodal-precision-testing` 负责用固定流程执行精度回归

## 版本说明

当前仓库内容面向 `0.18` 分支维护。与 `vllm`、`vllm-ascend`、Ascend 环境相关的默认参数和工作流，以该分支中的 skill 内容为准。
