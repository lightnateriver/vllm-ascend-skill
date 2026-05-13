# vllm-multimodal-evaluator

面向 stock `vllm` 或 `vllm-ascend` OpenAI 兼容服务的多模态能力评估 skill，目标是帮助使用者快速构造规则化测试数据，并用可复用 checklist 验证图片、视频、多图输入和图文穿插能力是否正常。

采用**两阶段测试**策略：
- **Phase 1 — 格式读取**：验证服务能否正常读取各格式媒体文件（HTTP 200 + 无错误）
- **Phase 2 — 语义理解**：验证模型是否正确理解媒体内容（关键词匹配）

这两个阶段覆盖三种媒体传输模式：`file://`、Base64 和 HTTP。

这个 skill 可以视作 `vllm-ascend-use` 的配套子 skill：

- `vllm-ascend-use` 负责通用部署、benchmark 和输入一致性验证
- `vllm-multimodal-evaluator` 负责测试数据生成和多模态能力支持矩阵验证

## 适用场景

这个 skill 适合下面几类任务：

- 需要生成简单、可控、可重复的本地图像测试数据
- 需要把 JPG 图片进一步生成低体积、多格式视频测试数据
- 需要启动 `Qwen3.5-4B` 并允许服务读取本地媒体目录
- 需要验证图片格式是否支持 `JPG`、`PNG`、`WebP`、`BMP`、`TIFF`
- 需要验证图片通过本地 `file://`、`Base64` 和 `HTTP` 三种请求是否正常
- 需要验证多图输入、多图顺序理解是否正常
- 需要验证文本和多图在消息内容中穿插排列时是否正常
- 需要验证视频格式和视频分辨率是否正常
- 需要区分"格式读取失败"和"模型理解错了"这两种不同的失败类型
- 需要产出机器可读和人可读的 `PASS` / `FAIL` 能力报告，报告头部包含服务配置信息

## 能力概览

本 skill 当前覆盖六个方向：

1. **测试数据生成**
   生成蓝色几何形状、绿色背景的图片数据，并按固定目录结构保存。
2. **视频测试数据生成**
   基于本地 JPG 测试图片生成低体积、多格式、多分辨率视频。
3. **多模态服务部署**
   提供一个通用的 vLLM 启动脚本，适用于 NVIDIA GPU 和 Ascend NPU，默认开启所有功能。
4. **格式读取测试**
   自动验证各格式媒体文件能否被服务正常读取，不校验语义内容。
5. **能力 checklist 执行**
   自动发起图片、视频、多图和图文穿插请求，支持 `file://` / Base64 / HTTP 三种传输模式，并输出 Markdown 与 JSON 报告。
6. **Function Calling 能力测试**
   使用预定义的15个测试用例，验证模型的函数名匹配、必选参数完整性、并行调用和多轮对话上下文保持能力。

## 默认测试策略

- 所有格式读取测试使用 `max_completion_tokens=16`（仅用于确认读取成功）。
- 所有语义理解测试使用 `max_completion_tokens=512`。
- Markdown 报告会记录每个 case 的完整 `/v1/chat/completions` 请求 Payload，包含文本输入、媒体 URL 或 Base64 数据、`model`、`temperature`、`stream` 和 `max_completion_tokens`。
- Markdown 报告会记录每个 case 的完整模型输出。
- JSON 报告会以结构化字段保存同样的完整请求输入和完整模型输出，便于后续程序化分析。
- 报告头部包含服务配置表，显示 dtype、chunked prefill、async scheduling、prefix caching、function calling 等启动参数。

## 判责边界

这个 skill 的价值不只是跑出 `PASS/FAIL`，而是把问题拆成不同层级：

- `Phase 1 ingestion FAIL` 优先按 `Engineering Error` 候选处理。
- `Phase 2 semantic FAIL` 且 ingestion 已通过时，优先按 `Model Capability Limitation` 候选处理。
- 输出没有稳定落到要求短格式时，应单列为 `Output Format / Protocol Issue`，不要直接算成视觉能力失败。

如果视频或 HTTP case 后续被证明是测试链路、静态服务或 timeout 策略问题，并且修复后 rerun 通过，那么历史失败不应继续计入模型错误。

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
    ├── start_qwen35_4b_vllm.sh          # 通用启动脚本
    ├── run_multimodal_capability_tests.py  # 两阶段测试主脚本
    ├── function_calling_test.json
    └── fc_test.py
```

## 推荐使用方式

### 1. 先确认目标

先确认当前任务是"多模态能力评估"，而不是：

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

如果服务还没起来，可以用通用的 vLLM 启动脚本（适用于 NVIDIA GPU 和 Ascend NPU）：

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
ALLOWED_LOCAL_MEDIA_PATH=/path/to/project \
PORT=8000 \
bash scripts/start_qwen35_4b_vllm.sh
```

这样服务就能读取 `pics/` 和 `video/` 下的本地媒体。
所有功能默认开启：chunked prefill / async scheduling / prefix caching / function calling。

如需测试 HTTP 模式，额外启动静态文件服务器：

```bash
python3 -m http.server 9000 --directory /path/to/project
```

建议在正式跑 capability 前，先直接 `curl` 一个图片 URL 和一个视频 URL，确认静态服务真的能把测试素材暴露出来。

### 4. 跑 checklist

基础用法（仅 file_url + base64）：

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B
```

完整用法（含 HTTP 模式和所有服务配置标志）：

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B \
  --dtype bfloat16 \
  --chunked-prefill True \
  --async-scheduling True \
  --prefix-caching True \
  --function-calling True \
  --media-base-url http://127.0.0.1:9000
```

默认会输出：

- `results/qwen35_multimodal_capability_report.md`
- `results/qwen35_multimodal_capability_report.json`

生成的 Markdown 报告包含五类信息：

1. **服务配置表** — 显示 dtype、chunked prefill、async scheduling、prefix caching、function calling 等启动参数
2. **媒体格式读取测试** — Phase 1 结果，验证各格式能否被服务正常读取
3. **语义理解测试** — Phase 2 结果，按能力项分类的详细结果
4. **语义理解汇总** — 所有语义测试的汇总矩阵
5. **失败 Case 明细** — 失败的 case 及其 HTTP 状态和输出摘要
6. **完整 Case 输入与输出** — 包含完整请求 Payload 和完整模型回答

JSON 报告建议优先消费结构化摘要字段，用于后续统一汇总：

- `counts_by_status`
- `counts_by_test_type`
- `engineering_errors`
- `model_limitations`
- `non_pass_cases`

### 5. 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dtype` | `bfloat16` | 模型精度 |
| `--chunked-prefill` | `True` | 是否启用 chunked prefill |
| `--async-scheduling` | `True` | 是否启用异步调度 |
| `--prefix-caching` | `True` | 是否启用 prefix caching |
| `--function-calling` | `True` | 是否启用 function calling |
| `--gpu-memory-utilization` | `0.7` | GPU 显存利用率 |
| `--enforce-eager` | `True` | 是否启用 eager mode |
| `--media-base-url` | `None` | HTTP 模式的媒体访问地址，如 `http://127.0.0.1:9000` |

## checklist 覆盖范围

当 `--media-base-url` 设置时，以下每项能力会同时按 file_url / base64 / http 三种模式测试：

| 能力项 | file_url | base64 | http |
|--------|:--------:|:------:|:----:|
| 图片单图格式（JPG/PNG/WebP/BMP/TIFF） | ✅ | ✅ | ✅ |
| 图片分辨率（256x512 / 720x1280 / 1920x1080） | ✅ | - | ✅ |
| 多图输入理解（7张图） | ✅ | ✅ | ✅ |
| 文本和多图穿插排列 | ✅ | - | - |
| 视频格式（MP4/AVI/MOV/MKV） | ✅ | - | ✅ |
| 视频分辨率（720x1280 / 1080x1920） | ✅ | - | ✅ |
| 视频首尾形状识别 + 全序列理解 | ✅ | - | ✅ |

## Function Calling 测试

### 前提条件

启动 vLLM 服务时需要开启 function calling 支持：

```bash
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

### 执行测试

```bash
python scripts/fc_test.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model /path/to/Qwen3.5-4B
```

`fc_test.py` 支持 `--endpoint`、`--model`、`--test-file` 三个参数。

### 测试用例

`scripts/function_calling_test.json` 包含15个标准测试用例，覆盖：

| 场景 | 用例数 | 说明 |
|------|:------:|------|
| 基础功能 | 4 | 单函数调用，必选参数完整 |
| 多工具并行 | 2 | 一次请求中并行调用多个函数 |
| 参数缺失 | 2 | 必选参数不完整时的模型行为 |
| 多轮对话 | 2 | 上下文关联的连续函数调用 |
| 无需调用 | 2 | 闲聊场景不触发函数调用 |
| 模糊参数 | 1 | 参数含歧义时的模型行为 |
| 非法参数 | 1 | 参数值非法时的模型行为 |
| 长文本干扰 | 1 | 无关长文本中提取函数调用 |

测试脚本采用宽松评估策略：

- **核心校验**：函数名匹配 + 必选参数是否存在（不做字符串值精确匹配）
- **参数缺失场景**：小模型倾向于填默认值而非追问，视为 PASS
- **非法参数场景**：小模型可能仍执行调用，视为 PASS
- **无需调用场景**：模型若仍调用函数则校验函数名和参数

### 结果解读

输出格式示例如下：

```
PASS | FC-001 [基础功能] 调用 get_weather({'city': '北京', 'date': '2025-01-01'})
PASS | FC-005 [多工具并行] 并行调用：['get_weather', 'calculate']
...
=== 汇总: 15/15 通过 ===
```

## 结果解读建议

当某个 case 失败时，先区分是格式读取问题还是语义理解问题（两阶段测试已自动分离）：

**格式读取失败**（Phase 1 FAIL）：
- `--allowed-local-media-path` 是否正确设置
- 静态文件服务器是否正在运行（HTTP 模式）
- 文件权限和路径是否正确
- vLLM 版本是否支持该格式

**语义理解失败**（Phase 2 FAIL）：
- 请求是否构造错误（Phase 1 已读通，这一步不太可能）
- 输出是否在 `512` token 上限下仍然被截断
- 模型是否真的理解错了

如果某个 case 的回答明显被截断，需要结合完整 Markdown 报告里的请求 Payload、完整输出和 JSON 里的原始结果判断问题来源，不要直接把它归类为媒体格式不支持。

如果 capability rerun 已经证明某个历史问题属于测试链路修复项，那么最终验收报告里应把它归为已解决工程问题，而不是继续累计为模型错误。

## 相关文件

- Skill 定义：`SKILL.md`
- 数据集规范：`references/dataset-layout.md`
- checklist 设计：`references/checklist-design.md`
- 图片生成：`scripts/generate_shape_dataset.py`
- 视频生成：`scripts/generate_shape_videos.py`
- 服务启动：`scripts/start_qwen35_4b_vllm.sh`
- 能力测试（两阶段）：`scripts/run_multimodal_capability_tests.py`
- Function Calling 测试用例：`scripts/function_calling_test.json`
- Function Calling 测试脚本：`scripts/fc_test.py`
