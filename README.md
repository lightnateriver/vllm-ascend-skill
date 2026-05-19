# vllm-ascend-skill

面向 `vllm` 与 `vllm-ascend` 的多 skill 仓库，当前聚焦 Ascend NPU 场景下的部署、能力验证、精度回归与 OpenAI API server 侧性能分析。

## 仓库内 skill

当前仓库包含四个并列 skill：

- `vllm-ascend-use`
  面向 stock `vllm-ascend` 的通用实战工作流，覆盖架构理解、服务部署、性能测试和前 `LLM` 输入一致性验证。
- `vllm-multimodal-evaluator`
  面向 stock `vllm` 或 `vllm-ascend` OpenAI 兼容服务的多模态能力评估工作流，覆盖规则化图片和视频测试数据生成，以及两阶段测试：格式读取和语义理解。当前标准能力项默认全量覆盖 `local_path` / `base64` / `http` 三种输入模式，并包含 `1~10` 多视频理解、大图 smoke 与 function calling 检查。
- `vllm-multimodal-precision-testing`
  面向本地 `vLLM` 或 `vllm-ascend` OpenAI 兼容服务的多模态精度回归工作流，覆盖 `L0`、`L0.5`、`MME`、`MMBench_DEV_EN`，标准流程默认对 `local_path`、`base64`、`http` 三种输入方式做全量测试与汇总。
- `vllm-ascend-api-server-profiler`
  面向 stock `vllm-ascend` OpenAI API server 的热点分析工作流，强调外置 monkey patch、请求级 profile、stage-based breakdown，以及基于热点函数内部逻辑的多方案调优分析。

## 仓库结构

```text
.
├── README.md
├── vllm-ascend-use/
├── vllm-multimodal-evaluator/
├── vllm-multimodal-precision-testing/
└── vllm-ascend-api-server-profiler/
```

## 推荐使用顺序

建议按下面顺序使用，而不是一上来直接跑所有精度项：

1. 先用 `vllm-ascend-use` 完成部署、探活和最小文本请求验证。
2. 再用 `vllm-multimodal-evaluator` 确认图片、视频、`local_path` / `base64` / `http` 链路是否真的可用，并区分链路问题与模型能力问题。
3. 最后再用 `vllm-multimodal-precision-testing` 跑 `L0`、`L0.5`、`MME`、`MMBench`。
4. 如果问题进入 API server 热点或时延归因阶段，再使用 `vllm-ascend-api-server-profiler`。

## 适用场景

适合下面几类需求：

- 需要解释 `vLLM V1`、`vllm-ascend` 与 Ascend NPU 的执行路径
- 需要快速部署一个 stock `vllm serve` 服务并跑通多模态请求
- 需要比较两次部署在 `LLM` 前的输入是否一致
- 需要生成规则化图像或视频测试数据，并对多模态服务做能力支持矩阵验证
- 需要在模型调优后执行固定冒烟、多图 `1~40` 回归，以及 `MME` / `MMBench` 回归测试
- 需要区分服务链路错误、输出格式问题和模型能力问题
- 需要分析 OpenAI API server 侧而不是 engine core / TP worker 侧的热点函数

## 仓库级失败归因口径

为了避免把链路问题误报成模型问题，仓库内文档和脚本统一按下面三类口径解释失败：

- `Engineering Error`
  服务未启动、HTTP 拉取失败、请求超时、媒体路径不可读、静态服务问题、测试脚本链路异常。
- `Model Capability Limitation`
  媒体读取成功，但语义理解、函数调用、多图检索、多视频顺序理解或复杂推理没有达到预期。
- `Output Format / Protocol Issue`
  模型有回答，但没有稳定收敛到要求的短答案、`yes/no`、选项字母或结构化函数调用格式。

如果某个问题后续被 capability rerun 证明是测试链路问题，并且修复后通过，那么历史失败不应继续计入模型错误。

## 当前 evaluator 版本边界

基于当前 `0.18` 分支的最新完整 evaluator 三模式复测，可以把能力边界概括成下面几点：

- 标准图片、标准视频、三种输入模式的读取链路已经稳定
- 标准单图、多图、图文穿插、普通视频格式与视频细节能力整体正常
- 多视频 `1~10` 在三种输入模式下呈现相似失败分布，主要应视为模型能力不稳定，而不是 `http` 或 `base64` 链路故障
- 大图 smoke 的读取链路正常，但大图语义 smoke 仍未稳定通过，主要应视为模型能力或输出收敛问题
- Function Calling 当前是部分通过，问题集中在函数选择和参数满足度，不是服务不可用

## 推荐最终产物

一次完整验收建议至少保留下面这些产物：

- 部署命令和部署日志
- `/v1/models` 与最小 `/v1/chat/completions` 探活结果
- capability JSON / Markdown 报告
- `L0` 摘要
- `L0.5` 的 `summary.json` / `summary.csv` 与逐 case JSON
- `MME` 的 score / pred 文件
- `MMBench` 的 acc / pred 文件
- 一份统一汇总 JSON / Markdown，明确区分：
  - `engineering_errors`
  - `model_limitations`
  - `output_format_or_protocol_issues`

## 安装示例

如果要从这个仓库安装多个 skill，可以使用：

```bash
python /root/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lightnateriver/vllm-ascend-skill \
  --path vllm-ascend-use vllm-multimodal-evaluator vllm-multimodal-precision-testing vllm-ascend-api-server-profiler \
  --ref 0.18
```

如果只安装其中一个，也可以只传单个 `--path`。

## 标准复测建议

如果任务是“部署后完整复测”，推荐使用 `vllm-multimodal-precision-testing/scripts/run_standard_retest.py` 作为仓库级标准入口。

它会按固定顺序执行：

1. `vllm-multimodal-evaluator` capability 检查
2. `vllm-multimodal-precision-testing` 的三模式全量 `L0/L0.5/MME/MMBench`

并默认把同一轮复测结果统一落到：

```text
vllm-multimodal-precision-testing/retest-runs/<run_name>/
```

其中同时保留：

- capability JSON / Markdown 报告
- precision 三模式汇总 JSON / Markdown
- 每个模式每个 step 的命令、标准输出和标准错误

## 版本说明

当前仓库内容面向 `0.18` 分支维护。与 `vllm`、`vllm-ascend`、Ascend 环境相关的默认参数和工作流，以该分支中的 skill 内容为准。
