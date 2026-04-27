# vllm-multimodal-precision-testing

面向本地 `vLLM` 或 `vllm-ascend` OpenAI 兼容服务的多模态精度回归 skill，目标是帮助使用者在模型调优、部署变更或参数调整后，用固定层级的回归流程快速确认多模态能力是否退化。

这个 skill 可以视作 `vllm-multimodal-evaluator` 的回归测试配套子 skill：

- `vllm-multimodal-evaluator` 负责生成规则化图片和视频测试数据，并验证服务的多模态能力支持矩阵
- `vllm-multimodal-precision-testing` 负责在已有服务和测试数据基础上，执行分层精度回归

## Skill 描述

这是一个面向多模态模型回归测试的标准 skill，支持对本地服务执行：

- `L0` 固定图片与视频冒烟测试
- `L0.5` 固定 `1~40` 张图多图精度测试
- `L1` 基于 `MME` 和 `MMBench_DEV_EN` 的更全面精度测试

它强调：

- 固定数据
- 固定提示词策略
- 固定抽取判分方式
- 固定执行顺序

这样在多轮调优之间更容易做横向对比。

## 功能说明

这个 skill 当前提供五类能力：

1. `L0` 多模态冒烟测试
   使用固定的 7 张图片和 1 个视频做短答案回归，快速捕获明显功能退化。
2. `L0.5` 多图精度测试
   使用内置的 `1~40` 张图数据集，检查单请求多图检索、目标定位和短答案输出稳定性，并保留逐 case 结果。
3. `MME` 本地精度测试
   使用官方 `MME.tsv` 执行 yes/no 抽取判分，并输出逐题结果与汇总分数。
4. `MMBench_DEV_EN` 本地精度测试
   使用官方 `MMBench_DEV_EN.tsv` 执行 MCQ 字母抽取与近似官方 heuristic 的 grouped scoring，并保留逐题结果。
5. 一键回归执行
   通过统一入口脚本，按 `L0 -> MME -> MMBench` 顺序执行，并输出一份最终总览结果，同时支持统一切换媒体输入方式。

## 新增能力

这次更新后，`L0`、`L0.5`、`MME` 和 `MMBench_DEV_EN` 都支持统一的三种媒体输入方式：

- `base64`
- `local_path`
- `http`

统一参数如下：

- `--media-mode`
  选择媒体输入方式。
- `--media-root`
  `local_path` / `http` 模式下的本地媒体根目录；当评测脚本需要把数据集里的 base64 素材落盘时，也会写到这里。
- `--media-base-url`
  `http` 模式下本地 HTTP 静态服务的基地址，例如 `http://127.0.0.1:9000`。

默认建议：

- `L0`、`MME`、`MMBench` 默认用 `local_path`
- `L1` 默认并发改为 `16`
- `http` 只建议使用本机静态服务，不建议依赖远端图床

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

### L0.5 Multi-Pics

目标是补上 `L0` 和 `L1` 之间的一层多图定位回归，重点观察：

- 单请求携带 `1~40` 张图片时的识别稳定性
- 目标颜色和形状的联合检索能力
- 第几张图定位能力
- 高图数场景下的顺序偏移和近邻误判

这套数据集采用规则化二维/三维几何图形，适合作为每次调优后的固定回归集。

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
├── multi-pics-datasets/
│   ├── README.md
│   ├── generate_dataset.py
│   └── cases/
└── scripts/
    ├── l0_multimodal_smoke.py
    ├── media_input_utils.py
    ├── multi_pics_eval.py
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
- 多图数据集目录：`multi-pics-datasets/cases`

同时，回归请求默认依赖：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

如果要测试 `local_path`，服务启动时还需要带上：

```bash
--allowed-local-media-path /your/media/root
```

如果要测试 `http`，建议在本机起一个静态文件服务，例如：

```bash
python3 -m http.server 9000 --directory /your/media/root
```

然后把评测参数里的：

- `--media-mode http`
- `--media-root /your/media/root`
- `--media-base-url http://127.0.0.1:9000`

配套传进去。

### 2. 先跑 L0 再跑 L1

推荐顺序：

1. 先跑 `L0`
2. 再跑 `L0.5` 多图测试
3. 再跑 `MME`
4. 最后跑 `MMBench_DEV_EN`

如果只是想一键执行，直接使用：

```bash
python scripts/run_full_regression.py
```

如果要统一切换为三种媒体输入之一，可以直接用：

```bash
python scripts/run_full_regression.py \
  --media-mode local_path \
  --media-root /mnt/sfs_turbo/codes/lzp/vllm-ascend-precision-testing/l0_assets
```

或者：

```bash
python scripts/run_full_regression.py \
  --media-mode http \
  --media-root /mnt/sfs_turbo/codes/lzp/vllm-ascend-precision-testing/l0_assets \
  --media-base-url http://127.0.0.1:9000
```

如果要单独跑多图测试，推荐使用：

```bash
python scripts/multi_pics_eval.py \
  --wait-ready \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --dataset-dir multi-pics-datasets/cases \
  --media-mode local_path \
  --json
```

这里建议始终带上 `--wait-ready`。因为真实实践里服务启动早期可能已经能连通 HTTP，但仍返回 `502`。这个脚本会先确认：

1. `/v1/models` 返回 `200`
2. 一个最小 text chat 请求也返回 `200`

只有两者都成功才开始正式评测，避免把服务启动期噪声误判成模型精度问题。

### 2.5. 三种媒体输入的使用建议

- `base64`
  兼容性最好，适合做默认兜底，也最接近公开 benchmark 常见发法。
- `local_path`
  最适合验证本地文件直传链路，要求服务允许本地媒体路径访问。
- `http`
  最适合模拟“请求里传 URL”场景，建议只用本机 `127.0.0.1` 静态服务，避免远端网络波动污染精度结果。

### 3. 保留逐题输出件

这个 skill 默认会保留明细结果，便于复盘：

- `L0`
  JSON 输出会保留每个固定 case 的请求结果和返回内容。
- `L0.5`
  `multi-pics-runs/<run_name>/summary.json` 会保留整轮汇总和全部 case 明细。
  `multi-pics-runs/<run_name>/summary.csv` 会保留每个 case 的计分结果。
  `multi-pics-runs/<run_name>/<case>.json` 会保留题目、标准答案、原始回答、抽取结果和最终状态。
- `MME`
  `*.pred.tsv` 保留每道题的 `question`、`answer`、`prediction`、`extracted`、`score`。
- `MMBench`
  `*.pred_all.tsv` 保留每道题和每个 circular 变体的 `question`、`answer`、`prediction`、`extracted`、`row_hit`。
  `*.pred.tsv` 保留最终计分主样本。

`MME` 和 `MMBench` 现在还会额外保留：

- `media_mode`
- `image_ref`
- `local_image_path`

这样后续分析时可以直接看出某一次分数对应的是 `base64`、`local_path` 还是 `http`。

### 4. 多图数据集设计

内置多图数据集位于 `multi-pics-datasets/`，是一个确定性的 `1~40` 张图回归集：

- `01` 目录只有 `1` 张图，题目是封闭式 `YES` / `NO`
- `02` 到 `40` 目录分别包含 `N` 张图，题目是“第几张图是某种颜色某种形状”，只允许回答 `1` 个数字
- 同一 case 内，颜色和形状组合唯一
- 同一 case 内允许形状重复，但重复形状的颜色不会重复
- 每个 case 都提供 `question.md`、`answer.md`、`answer.json` 和对应图片

如果要重建这套数据集，可以执行：

```bash
python multi-pics-datasets/generate_dataset.py
```

这套数据集适合观察高图数条件下的真实能力边界。以当前 stock `Qwen3.5-4B` 在原生 `vllm-ascend` 服务上的一次完整测试为例，曾得到：

- `total=40`
- `correct=24`
- `wrong=16`
- `unknown=0`
- `timeout=0`

这个例子说明，修复启动期误判后，多图测试能更干净地区分“服务异常”与“模型能力边界”。

## 三种输入模式的参考结果

下面是这次对 stock `Qwen3.5-4B` 的一组参考结果，可作为后续回归的对照基线：

### L0

- `base64`: `10/10`
- `local_path`: `10/10`
- `http`: `10/10`

### L0.5

- `base64`: `24/40`, `accuracy=0.6000`
- `local_path`: `24/40`, `accuracy=0.6000`
- `http`: `24/40`, `accuracy=0.6000`

### MME

- `base64`: `80.6234%`
- `local_path`: `80.7077%`
- `http`: `80.6655%`

### MMBench_DEV_EN

- `base64`: `82.5601%`
- `local_path`: `82.9038%`
- `http`: `82.7320%`

从这组结果看，三种输入方式的精度差异很小，整体符合预期。后续若出现大幅偏移，更可能是输入链路或服务处理逻辑发生了变化。

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
- `L0` 资源：`assets/l0/`
- 多图数据集：`multi-pics-datasets/`
- `L0` 冒烟测试：`scripts/l0_multimodal_smoke.py`
- 多图精度测试：`scripts/multi_pics_eval.py`
- `MME` 回归测试：`scripts/mme_eval_local.py`
- `MMBench_DEV_EN` 回归测试：`scripts/mmbench_eval_local.py`
- 一键回归入口：`scripts/run_full_regression.py`
