# vllm-multimodal-evaluator

面向 stock `vllm` 或 `vllm-ascend` OpenAI 兼容服务的多模态能力验证 skill。

这个 skill 的定位很明确：

- 负责“服务是否支持某类多模态输入”和“模型是否具备基础能力”的能力验证
- 负责生成规则化图片和视频测试数据
- 负责把失败分成链路问题、输出格式问题、模型能力问题
- 不负责 `L0`、`L0.5`、`MME`、`MMBench` 这类精度回归

精度回归继续交给 `vllm-multimodal-precision-testing`。

## 这个 skill 解决什么问题

当用户说“这个模型能不能吃图片、视频、多图、图文穿插、HTTP URL、本地文件路径、大图、function calling”，优先应该用这个 skill。

它更像一个“能力支持矩阵”和“链路体检工具”，目的不是算 benchmark 分数，而是回答下面这些问题：

- 服务能不能读取图片和视频
- `local_path`、`base64`、`http` 三种输入方式是否都能工作
- 图片、多图、图文穿插、视频语义理解是否基本正常
- 大图输入是否至少能做 smoke 级验证
- function calling 链路是否正常
- 某个失败到底是媒体链路错、抽取协议错，还是模型真的不会

## 职责边界

### `vllm-multimodal-evaluator` 负责

| 能力方向 | 默认是否开启 | 目的 |
| --- | :---: | --- |
| 规则化图片数据生成 | 是 | 生成稳定的单图、多图、大图测试素材 |
| 规则化视频数据生成 | 是 | 生成标准视频和大尺寸 MP4 视频素材 |
| Phase 1 格式读取检查 | 是 | 先判断媒体是否能被服务正确读取 |
| Phase 2 语义理解检查 | 是 | 再判断模型是否正确理解媒体内容 |
| 三种输入模式能力矩阵 | 是 | 同时验证 `local_path/base64/http` |
| 多视频理解 `1~10` | 是 | 验证单请求内多视频顺序理解阈值 |
| 大图 smoke suite | 是 | 默认验证 `4096x4096`、`4096x6144`、`4096x8192` |
| Function Calling 能力检查 | 是 | 默认纳入标准 capability run |

### `vllm-multimodal-precision-testing` 负责

- `L0`
- `L0.5`
- `MME`
- `MMBench_DEV_EN`
- 三种输入模式下的精度回归
- runner 自检和标准复测编排

## 两阶段设计

这个 skill 采用两阶段测试策略：

- `Phase 1 - 格式读取`
  目标是证明服务真的读到了媒体，而不是把“媒体都没读到”误记成模型不懂。
- `Phase 2 - 语义理解`
  目标是证明媒体已可达的前提下，模型是否正确理解图像或视频内容。

这两个阶段分开后，失败可以更稳定地归因：

- `Phase 1 FAIL` 优先归为链路、路径、权限、静态服务、格式支持问题
- `Phase 2 FAIL` 且对应 ingestion 已通过时，优先归为模型能力问题
- 模型有输出但不按要求收敛到短答案、固定格式时，归为输出格式或协议问题

## 统一输入口径

对外统一只使用三种 transport 名称：

- `local_path`
- `base64`
- `http`

其中 `local_path` 的底层实现仍然是 `file://` URL，所以报告中会同时保留：

- `transport_mode=local_path`
- `transport_impl=file_url`

这样对用户口径统一，对工程排查也保留足够细节。

## 默认测试内容

标准 capability run 默认覆盖下面这些测试项。

### Phase 1

| 测试项 | local_path | base64 | http | 目的 |
| --- | :---: | :---: | :---: | --- |
| 图片格式读取 | ✅ | ✅ | ✅ | 验证 jpg/png/webp/bmp/tiff 是否都能读 |
| 图片分辨率读取 | ✅ | ✅ | ✅ | 验证不同分辨率图片读入 |
| 视频格式读取 | ✅ | ✅ | ✅ | 验证 mp4/avi/mov/mkv 是否可读 |
| 视频分辨率读取 | ✅ | ✅ | ✅ | 验证不同分辨率视频读入 |

### Phase 2

| 测试项 | local_path | base64 | http | 目的 |
| --- | :---: | :---: | :---: | --- |
| 图片单图语义理解 | ✅ | ✅ | ✅ | 验证单图识别与颜色/背景理解 |
| 图片分辨率语义理解 | ✅ | ✅ | ✅ | 验证高低分辨率图片理解一致性 |
| 多图输入理解 | ✅ | ✅ | ✅ | 验证多图顺序和图形列表输出 |
| 图文穿插输入理解 | ✅ | ✅ | ✅ | 验证 interleave 内容顺序理解 |
| 视频语义理解 | ✅ | ✅ | ✅ | 验证视频基本理解 |
| 视频分辨率语义理解 | ✅ | ✅ | ✅ | 验证不同视频分辨率理解 |
| 视频细节与顺序理解 | ✅ | ✅ | ✅ | 验证 first/last/order 类问题 |
| 多视频输入理解 `1~10` | ✅ | ✅ | ✅ | 验证单请求内 1 到 10 个视频的顺序理解 |

### 默认开启的扩展 suite

| Suite | 默认 | 目的 |
| --- | :---: | --- |
| `phase2_semantic_multi_video_standard` | 开 | 验证 `1~10` 多视频输入理解 |
| `large_image_smoke` | 开 | 验证 4K 级图片读取和基础语义 smoke |
| `phase2_function_calling_standard` | 开 | 验证 function calling 能力和输出链路 |

大图 smoke 默认覆盖：

- `4096x4096`
- `4096x6144`
- `4096x8192`

## 数据集与素材

### 图片目录

```text
pics/<resolution>/<format>/<shape>.<ext>
```

### 视频目录

```text
video/<resolution>/<format>/shapes.<ext>
```

用于多视频理解的单形状标准视频额外放在：

```text
video/720x1280/mp4/<shape>.mp4
```

### 内置图像分辨率

- `standard`
  - `256x512`
  - `720x1280`
  - `1920x1080`
- `large`
  - `4096x4096`
  - `4096x6144`
  - `4096x8192`

### 内置视频分辨率

- `standard`
  - `720x1280`
  - `1080x1920`
- `large`
  - `4096x4096`
  - `4096x6144`
  - `4096x8192`

大尺寸视频默认只生成 `mp4`，避免素材成本过高。

## 常用命令

### 生成图片

```bash
python scripts/generate_shape_dataset.py
```

仅生成标准分辨率：

```bash
python scripts/generate_shape_dataset.py --resolution-profile standard
```

仅生成大图：

```bash
python scripts/generate_shape_dataset.py --resolution-profile large
```

### 生成视频

```bash
python scripts/generate_shape_videos.py
```

标准分辨率生成会同时产出：

- `shapes.mp4` 顺序视频
- `square.mp4` 到 `cube.mp4` 单形状视频

仅生成大尺寸 MP4 视频：

```bash
python scripts/generate_shape_videos.py --resolution-profile large
```

### 标准 capability run

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B \
  --media-base-url http://127.0.0.1:9000 \
  --auto-start-media-server
```

关闭大图 smoke：

```bash
python scripts/run_multimodal_capability_tests.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3.5-4B \
  --media-base-url http://127.0.0.1:9000 \
  --no-large-image-smoke
```

只测部分 transport：

```bash
python scripts/run_multimodal_capability_tests.py \
  --transport-modes local_path http
```

### 单独跑 function calling

```bash
python scripts/fc_test.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model /path/to/Qwen3.5-4B
```

## 关键 CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--transport-modes` | `local_path base64 http` | 标准能力测试的 transport 矩阵 |
| `--include-large-image-smoke` | `True` | 默认启用大图 smoke suite |
| `--no-large-image-smoke` | `False` | 显式关闭大图 smoke |
| `--large-image-resolutions` | `4096x4096 4096x6144 4096x8192` | 大图 smoke 使用的分辨率列表 |
| `--include-function-calling` | `True` | 默认将 FC 测试并入 capability run |
| `--no-function-calling` | `False` | 显式关闭 FC suite |
| `--function-calling-test-file` | `scripts/function_calling_test.json` | 覆盖默认 FC 用例文件 |
| `--media-base-url` | `http://127.0.0.1:9000` | `http` 模式媒体静态服务地址 |
| `--auto-start-media-server` | `True` | 默认自动拉起本地静态媒体服务 |

## 输出与报告

JSON 和 Markdown 报告会明确体现：

- 服务配置
- transport modes
- enabled optional suites
- included test items
- counts by suite
- counts by media scale
- counts by transport mode
- 每个 case 的请求内容
- 每个 case 的模型输出
- 每个失败项的归因

关键字段：

| 字段 | 说明 |
| --- | --- |
| `suite_name` | 当前 case 属于哪个 suite |
| `media_scale` | `standard` 或 `large` |
| `transport_mode` | `local_path/base64/http` |
| `transport_impl` | `file_url/data_url/http_url` |
| `enabled_optional_suites` | 这次 run 打开的可选 suite |
| `included_test_items` | 报告中按 suite/category/test_type 聚合的覆盖清单 |
| `counts_by_suite` | 按 suite 聚合的 PASS/FAIL/BLOCKED/SKIP |
| `counts_by_media_scale` | 按标准/大尺寸聚合 |
| `counts_by_transport_mode` | 按三种输入方式聚合 |
| `failure_class` | 失败归类 |
| `root_cause_note` | 失败原因说明 |

## 当前默认版本验证结论

基于当前仓库版本的完整三模式 evaluator 复测，建议按下面口径理解结果：

- `Phase 1` 标准读取和大图读取已在 `local_path/base64/http` 下全通过
- 标准图片、多图、图文穿插、标准视频格式与细节能力整体正常
- 多视频 `1~10` 在三种模式下都表现出明显不稳定，主要属于模型能力问题，不是传输模式错误
- 大图语义 smoke 在三种模式下都可读但都未稳定通过，主要属于模型能力或输出不收敛问题
- Function Calling 当前是部分通过，不是服务阻塞型错误

## 失败归因规则

- `Phase 1 FAIL`
  优先看 `--allowed-local-media-path`、HTTP 静态服务、路径、权限、媒体格式支持。
- `Phase 2 FAIL`
  且对应 ingestion 已通过时，优先看模型能力。
- `HTTP` 失败
  不应直接记成模型错，要先排查静态媒体服务和 URL 构造。
- `explanation-heavy` 输出
  如果模型答了很多解释但没有按要求收敛到短格式，应标成 `Output Format / Protocol Issue`。

## 建议使用顺序

推荐顺序是：

1. 先用 `vllm-ascend-use` 把服务拉起来并确认 `/v1/models`、`/v1/chat/completions` 正常
2. 再用 `vllm-multimodal-evaluator` 验证能力支持矩阵
3. 如果能力链路正常，再进入 `vllm-multimodal-precision-testing`

这样可以把“服务不通”和“模型能力差”分开。
