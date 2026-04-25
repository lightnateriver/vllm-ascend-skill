# vllm-multimodal-precision-testing

面向本地 `vLLM` 或 `vllm-ascend` OpenAI 兼容服务的多模态精度回归 skill，目标是帮助使用者在模型调优、部署变更或参数调整后，用固定层级的回归流程快速确认多模态能力是否退化。

这个 skill 可以视作 `vllm-multimodal-evaluator` 的回归测试配套子 skill：

- `vllm-multimodal-evaluator` 负责生成规则化图片和视频测试数据，并验证服务的多模态能力支持矩阵
- `vllm-multimodal-precision-testing` 负责在已有服务和测试数据基础上，执行分层精度回归

## Skill 描述

这是一个面向多模态模型回归测试的标准 skill，支持对本地服务执行：

- `L0` 固定图片与视频冒烟测试
- `L1` 基于 `MME` 和 `MMBench_DEV_EN` 的更全面精度测试

它强调：

- 固定数据
- 固定提示词策略
- 固定抽取判分方式
- 固定执行顺序

这样在多轮调优之间更容易做横向对比。

## 功能说明

这个 skill 当前提供四类能力：

1. `L0` 多模态冒烟测试
   使用固定的 7 张图片和 1 个视频做短答案回归，快速捕获明显功能退化。
2. `MME` 本地精度测试
   使用官方 `MME.tsv` 执行 yes/no 抽取判分，并输出逐题结果与汇总分数。
3. `MMBench_DEV_EN` 本地精度测试
   使用官方 `MMBench_DEV_EN.tsv` 执行 MCQ 字母抽取与近似官方 heuristic 的 grouped scoring，并保留逐题结果。
4. 一键回归执行
   通过统一入口脚本，按 `L0 -> MME -> MMBench` 顺序执行，并输出一份最终总览结果。

此外，这个 skill 现在直接内置了 `L0` 的测试图片和视频资源，不再要求使用者额外准备固定几何图形素材目录。

## 目标用途

这个 skill 适合下面几类场景：

- 每次模型调优后需要快速确认多模态能力是否仍然正常
- 每次部署参数变化后需要做标准化回归
- 需要把“服务能不能工作”和“整体精度有没有退化”分成不同层级检查
- 需要保存逐题问题、标准答案、模型回答和抽取结果，便于后续分析

它不替代：

- 通用 Ascend 服务部署与 benchmark：交给 `vllm-ascend-use`
- API server 热点定位与剖析：交给 `vllm-ascend-api-server-profiler`
- 多模态测试数据生成与支持矩阵验证：交给 `vllm-multimodal-evaluator`

## 能力概览

### L0

目标是快速发现：

- 单图识别失效
- 多图顺序理解错误
- 短约束回答失效
- 基本视频理解异常

### L1 MME

目标是覆盖更广的 yes/no 感知与推理能力，例如：

- OCR
- counting
- existence
- position
- commonsense reasoning
- numerical calculation
- text translation

### L1 MMBench

目标是覆盖更广的 MCQ 感知与推理能力，例如：

- coarse perception
- fine-grained perception
- attribute / relation / logic reasoning
- OCR
- localization
- spatial relationship
- future prediction

## 目录结构

```text
.
├── README.md
├── SKILL.md
├── agents/
│   └── openai.yaml
├── assets/
│   └── l0/
│       ├── pics/720x1280/jpg/
│       └── video/720x1280/mp4/
└── scripts/
    ├── l0_multimodal_smoke.py
    ├── mme_eval_local.py
    ├── mmbench_eval_local.py
    └── run_full_regression.py
```

## 推荐使用方式

### 1. 先确认前置条件

要求本地服务已经可用，并能通过 `/v1/chat/completions` 访问。

默认假设：

- host: `http://127.0.0.1:8000`
- model: `/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B`
- `L0` 图片目录：`assets/l0/pics/720x1280/jpg`
- `L0` 视频路径：`assets/l0/video/720x1280/mp4/shapes.mp4`

同时，回归请求默认依赖：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

### 2. 先跑 L0 再跑 L1

推荐顺序：

1. 先跑 `L0`
2. 再跑 `MME`
3. 最后跑 `MMBench_DEV_EN`

如果只是想一键执行，直接使用：

```bash
python scripts/run_full_regression.py
```

对于 `L0`，脚本默认直接使用 skill 自带媒体资源：

- `circle.jpg`
- `cube.jpg`
- `cylinder.jpg`
- `rectangle.jpg`
- `rhombus.jpg`
- `square.jpg`
- `triangle.jpg`
- `shapes.mp4`

### 3. 保留逐题输出件

这个 skill 默认会保留明细结果，便于复盘：

- `L0`
  JSON 输出会保留每个固定 case 的请求结果和返回内容。
- `MME`
  `*.pred.tsv` 保留每道题的 `question`、`answer`、`prediction`、`extracted`、`score`。
- `MMBench`
  `*.pred_all.tsv` 保留每道题和每个 circular 变体的 `question`、`answer`、`prediction`、`extracted`、`row_hit`。
  `*.pred.tsv` 保留最终计分主样本。

## 安装与集成

该 skill 作为 `vllm-ascend-skill` 仓库中的标准子 skill，可以通过仓库路径直接安装。

只安装这个 skill：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-multimodal-precision-testing \
  --ref 0.18
```

与其他子 skill 一起安装：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-ascend-use vllm-multimodal-evaluator vllm-multimodal-precision-testing vllm-ascend-api-server-profiler \
  --ref 0.18
```

## 相关文件

- Skill 定义：`SKILL.md`
- UI 元信息：`agents/openai.yaml`
- `L0` 资产目录：`assets/l0/`
- `L0` 冒烟测试：`scripts/l0_multimodal_smoke.py`
- `MME` 回归测试：`scripts/mme_eval_local.py`
- `MMBench_DEV_EN` 回归测试：`scripts/mmbench_eval_local.py`
- 一键回归入口：`scripts/run_full_regression.py`
