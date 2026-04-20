# vllm-multimodal-evaluator

面向 stock `vllm` 或 `vllm-ascend` OpenAI 兼容服务的多模态能力评估 skill，目标是帮助使用者快速构造规则化测试数据，并用可复用 checklist 验证图片、视频、多图输入和图文穿插能力是否正常。

这个 skill 可以视作 `vllm-ascend-use` 的配套子 skill：

- `vllm-ascend-use` 负责通用部署、benchmark 和输入一致性验证
- `vllm-multimodal-evaluator` 负责测试数据生成和多模态能力支持矩阵验证

## 适用场景

这个 skill 适合下面几类任务：

- 需要生成简单、可控、可重复的本地图像测试数据
- 需要把 JPG 图片进一步生成低体积、多格式视频测试数据
- 需要启动 `Qwen3.5-4B` 并允许服务读取本地媒体目录
- 需要验证图片格式是否支持 `JPG`、`PNG`、`WebP`、`BMP`、`TIFF`
- 需要验证图片通过本地 `file://` 和 `Base64` 请求是否正常
- 需要验证多图输入、多图顺序理解是否正常
- 需要验证文本和多图在消息内容中穿插排列时是否正常
- 需要验证视频格式和视频分辨率是否正常
- 需要产出机器可读和人可读的 `PASS` / `FAIL` 能力报告

## 能力概览

本 skill 当前覆盖四个方向：

1. 测试数据生成
   生成蓝色几何形状、绿色背景的图片数据，并按固定目录结构保存。
2. 视频测试数据生成
   基于本地 JPG 测试图片生成低体积、多格式、多分辨率视频。
3. 多模态服务部署
   提供一个偏实战的 `Qwen3.5-4B` Ascend 启动脚本，用于本地媒体能力测试。
4. 能力 checklist 执行
   自动发起图片、视频、多图和图文穿插请求，并输出 Markdown 与 JSON 报告。

## 默认测试策略

- 所有 checklist 请求默认使用 `max_completion_tokens=512`。
- 如果单个 case 未显式覆盖输出 token，上限仍然是 `512`。
- Markdown 报告会记录每个 case 的完整 `/v1/chat/completions` 请求 Payload，包含文本输入、媒体 URL 或 Base64 数据、`model`、`temperature`、`stream` 和 `max_completion_tokens`。
- Markdown 报告会记录每个 case 的完整模型输出，不再只保留截断预览。
- JSON 报告会以结构化字段保存同样的完整请求输入和完整模型输出，便于后续程序化分析。

## 目录结构

```text
.
├── README.md
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── dataset-layout.md
│   └── checklist-design.md
└── scripts/
    ├── generate_shape_dataset.py
    ├── generate_shape_videos.py
    ├── start_qwen35_4b_vllm_ascend.sh
    └── run_multimodal_capability_tests.py
```

## 推荐使用方式

### 1. 先确认目标

先确认当前任务是“多模态能力评估”，而不是：

- 通用 Ascend 服务部署
- 大规模 benchmark
- 前 `LLM` 输入一致性验证
- API server 热点分析

如果任务更偏这些方向，优先改用：

- `../vllm-ascend-use/`
- `../vllm-ascend-api-server-profiler/`

### 2. 生成测试数据

生成图片：

```bash
python scripts/generate_shape_dataset.py
```

生成视频：

```bash
python scripts/generate_shape_videos.py
```

默认输出结构：

```text
pics/<resolution>/<format>/<shape>.<ext>
video/<resolution>/<format>/shapes.<ext>
```

### 3. 启动服务

如果服务还没起来，可以用：

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
ALLOWED_LOCAL_MEDIA_PATH=/path/to/project \
PORT=8000 \
scripts/start_qwen35_4b_vllm_ascend.sh
```

这样服务就能读取 `pics/` 和 `video/` 下的本地媒体。

### 4. 跑 checklist

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B
```

默认会输出：

- `results/qwen35_multimodal_capability_report.md`
- `results/qwen35_multimodal_capability_report.json`

默认请求输出 token 上限为 `512`。如需临时调整默认值，可以传入：

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B \
  --max-tokens 512
```

生成的 Markdown 报告包含三类信息：

- 能力项汇总表，显示每个 case 的输入文件、输出 token 上限、HTTP 状态、PASS/FAIL 和耗时
- 失败 case 明细
- 完整 Case 输入与输出，包含完整请求 Payload 和完整模型回答

## checklist 覆盖范围

当前脚本默认覆盖这些能力项：

- 图片单图格式支持：本地文件路径
- 图片单图格式支持：Base64
- 图片分辨率支持
- 多图输入理解
- 文本和多图穿插排列
- 视频格式支持
- 视频分辨率支持
- 视频理解细节

## 结果解读建议

当某个 case 失败时，不要第一时间判断成“模型不支持”。建议先区分：

- 请求是否构造错误
- 本地媒体路径是否未放开
- 服务是否成功接收媒体
- 输出是否在 `512` token 上限下仍然被截断
- 模型是否真的理解错了

这一步在多图和视频 case 里尤其重要，因为它们比单图 case 更容易出现“答案接近正确但被截断”的假阴性。

如果某个 case 的回答明显被截断，需要结合完整 Markdown 报告里的请求 Payload、完整输出和 JSON 里的原始结果判断问题来源，不要直接把它归类为媒体格式不支持。

## 相关文件

- Skill 定义：`SKILL.md`
- 数据集规范：`references/dataset-layout.md`
- checklist 设计：`references/checklist-design.md`
- 图片生成：`scripts/generate_shape_dataset.py`
- 视频生成：`scripts/generate_shape_videos.py`
- 服务启动：`scripts/start_qwen35_4b_vllm_ascend.sh`
- 能力测试：`scripts/run_multimodal_capability_tests.py`
