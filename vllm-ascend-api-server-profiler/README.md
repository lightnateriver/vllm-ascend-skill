# vllm-ascend-api-server-profiler

面向 stock `vllm-ascend` OpenAI API server 的热点分析 skill，目标是帮助使用者在不修改 `vllm` / `vllm-ascend` 源码的前提下，定位 API server 侧瓶颈并给出后续优化方向。

当前版本已经把一次更准确的 API server 侧复盘经验固化进 skill：先看顺序化的 stage，再看并发窗口，最后再下沉到叶子函数，而不是直接盯总包围函数。

## 适用场景

这个 skill 适合下面几类任务：

- 需要复现一轮 Qwen 或 Qwen3.5 的 API server profile
- 需要为 API server 侧生成请求级 timeline 和热点函数表
- 需要把 API server 预处理拆成顺序 stage，并解释 stage gap
- 需要解释并发函数的 `aggregate-total` 与 `window wall` 为什么不同
- 需要解释 `HTTP total wall time`、`API server CPU time` 和 `API server preprocess chain`
- 需要说明为什么函数时间不能直接相加
- 需要只分析 API server 侧，而不把 engine core / TP worker 当成主结论

## 核心约束

默认遵守这些限制：

- 不修改 `vllm` 源码
- 不修改 `vllm-ascend` 源码
- 不使用 `0407` patch
- 不使用 phase 方案
- 所有采集、打印与输出能力都通过外置 monkey patch 实现

## 目录结构

```text
.
├── README.md
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── metric-definitions.md
│   └── qwen35-stock-config.md
└── scripts/
    ├── start_profiled_api_server.sh
    ├── run_vllm_api_server_profiled.py
    ├── run_profiled_dataset_rounds.py
    └── summarize_api_server_profile.py
```

## 工作流概览

### 1. 准备输入

通常使用：

- `10k` 文本 token
- `40` 张图片
- `288x512`
- `3 warmup + 1 profile`
- 组内组外无重复文本与图像

### 2. 启动 profiled API server

使用：

```bash
scripts/start_profiled_api_server.sh
```

该脚本会通过外置入口 `run_vllm_api_server_profiled.py` 安装 monkey patch，并把 profile 结果写到指定目录。

### 3. 驱动 profile round

使用：

```bash
python scripts/run_profiled_dataset_rounds.py ...
```

对 profile round 单独打 profiling header。

### 4. 汇总结果

使用：

```bash
python scripts/summarize_api_server_profile.py \
  --functions-json <functions.json> \
  --driver-results <driver-results.json>
```

输出内容会覆盖：

- `stage_spans` 主视图
- `concurrency_windows` 并发窗口
- API server 时间线
- 热点函数表
- 总耗时口径解释
- 不能直接相加的原因
- 不改源码前提下的优化建议

## 推荐解读顺序

建议按下面顺序看结果：

1. 先看 `api_server_total.preprocess_wall` 和各个 `stage_spans`
2. 再看 `MediaConnector.load_from_url_async`、`SingleWriterShmObjectStorage.copy_to_buffer` 这类函数的 `concurrency_windows`
3. 最后看叶子热点函数，例如 `_preprocess`、`get_mm_hashes`、`Qwen2TokenizerFast.__call__`

这样做的原因是：

- stage 先回答“顺序关键路径卡在哪一段”
- concurrency window 再回答“多次调用是重叠、拉长窗口，还是纯工作量大”
- 叶子函数最后回答“具体该优化哪个功能点”

如果 `sum(stage wall)` 小于 `api_server_total.preprocess_wall`，通常差值应该解释为 stage 之间的 glue time，而不是遗漏的大函数。

## 建议优先关注的热点

在 Qwen3.5 的典型多模态场景中，通常更值得先盯这些 API server 侧函数：

- `Qwen2VLImageProcessorFast._preprocess`
- `ProcessorInputs.get_mm_hashes`
- `SingleWriterShmObjectStorage.copy_to_buffer`
- `Qwen2TokenizerFast.__call__`

像 `create_chat_completion` 这一类大包围函数，不适合作为最终优化结论。
