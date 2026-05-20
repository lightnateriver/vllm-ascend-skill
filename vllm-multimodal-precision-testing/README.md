# vllm-multimodal-precision-testing

面向本地 `vLLM` 或 `vllm-ascend` OpenAI 兼容服务的多模态精度回归 skill。

它的定位很简单：

- 在服务和基础能力都已经可用后，检查多模态精度是否退化
- 用固定层级的回归流程对比不同版本、不同部署参数、不同输入模式
- 把失败明确拆成工程问题、协议问题和模型能力问题

它和 `vllm-multimodal-evaluator` 是配套关系：

- `vllm-multimodal-evaluator` 负责验证能力支持矩阵和基础链路
- `vllm-multimodal-precision-testing` 负责做分层精度回归

## 这个 skill 负责什么

这个 skill 当前覆盖 5 类能力：

1. `L0`
   固定图片与视频 smoke 测试，快速发现最明显的多模态退化。
2. `L0.5`
   固定 `1~40` 张图的多图精度测试，检查高图数场景的定位和顺序稳定性。
3. `MME`
   官方 `MME.tsv` yes/no 精度测试，保留逐题抽取结果和原始分数。
4. `MMBench_DEV_EN`
   官方 `MMBench_DEV_EN.tsv` MCQ 精度测试，保留逐题抽取结果、`Z` 回退、逐类分数。
5. 运行时自检和复测编排
   默认跑三模式全量回归，同时跑 transport consistency 和 output contract self-check。

## 默认测试项

标准一键回归默认会执行以下内容：

- `L0`
- `L0.5`
- `MME`
- `MMBench`
- `transport_consistency_check.py`
- `output_contract_self_check.py`

默认输入模式也全开：

- `base64`
- `local_path`
- `http`

## 每个测试项的目的

### L0

目标是快速发现：

- 单图识别失效
- 多图顺序理解错误
- 短约束回答失效
- 基本视频理解异常

### L0.5

目标是验证单请求内多图能力，重点看：

- 1 到 40 张图时的检索稳定性
- 第几张图定位能力
- 图形和颜色联合检索能力
- 高图数下的顺序偏移和近邻误判

### MME

目标是覆盖更宽的 yes/no 感知与推理能力，例如：

- OCR
- counting
- existence
- position
- commonsense reasoning
- numerical calculation
- text translation

### MMBench

目标是覆盖更宽的 MCQ 感知与推理能力，例如：

- coarse perception
- fine-grained perception
- attribute reasoning
- relation reasoning
- logic reasoning
- OCR
- localization
- spatial relationship
- future prediction

### transport_consistency

目标是确认 `base64`、`local_path`、`http` 三种输入方式下，基础 L0 结果一致，不要把某一种输入模式的链路问题误判为模型退化。

### output_contract

目标是确认各脚本 `--json` 输出结构稳定，避免上层自动化因字段漂移失效。

## 输入模式说明

标准回归默认对三种输入模式全量执行：

| 模式 | 说明 | 备注 |
| --- | --- | --- |
| `base64` | `data:` URL | 默认兼容性最好 |
| `local_path` | `file://` URL | 对外统一口径，底层仍是文件路径 URL |
| `http` | 本机静态媒体 URL | 用来模拟 URL 输入场景 |

默认假设：

- host: `http://127.0.0.1:8000`
- model: `/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B`
- `L0` 图片目录：`../vllm-multimodal-evaluator/pics/720x1280/jpg`
- `L0` 视频路径：`../vllm-multimodal-evaluator/video/720x1280/mp4/shapes.mp4`
- 多图数据集目录：`multi-pics-datasets/cases`

## 默认执行顺序

标准一键回归默认按下面顺序执行：

1. `L0`
2. `L0.5`
3. `MME`
4. `MMBench`
5. `transport_consistency_check.py`
6. `output_contract_self_check.py`

同时对 `base64`、`local_path`、`http` 三种模式分别跑完整套流程。

## 判责原则

失败不应该只写“wrong”。

建议按三类解释：

- `Engineering Error`
  服务未就绪、HTTP 请求失败、媒体链路异常、超时、脚本或环境问题。
- `Model Capability Limitation`
  媒体成功读到了，但答案本身不对。
- `Output Format / Protocol Issue`
  模型有回答，但没有稳定落到短答案、`yes/no` 或选项字母要求。

### 典型判责方式

- `L0` 失败先看是不是链路问题，再看是不是模型不会
- `L0.5` 的 `wrong_answer` 多半是模型能力问题
- `MME` 的 `unknown` 不一定是模型不会，也可能是解释太长、抽取失败或格式漂移
- `MMBench` 的 `Z` 回退通常优先看输出协议问题
- `transport_consistency` 失败优先看静态媒体服务和 URL 构造

现在这些脚本还会先读取 `/v1/models`，并把传入的 `--model` 自动归一化到服务真实暴露的模型 id。
如果服务返回的模型 id 带尾部 `/`，不需要再手工改命令行参数。

## 报告里应该写什么

每份标准报告都应该明确包含：

- 三种输入模式是否都执行了
- `L0/L0.5/MME/MMBench` 的逐项结果
- `transport_consistency` 和 `output_contract` 的结果
- 每个失败项的归因
- 哪些失败是工程问题，哪些是模型能力问题，哪些是输出协议问题

建议字段：

- `requested_media_modes`
- `mode_comparison`
- `global_checks`
- `global_check_summary`
- `precision_summary`
- `engineering_errors`
- `model_limitations`
- `format_or_protocol_issues`
- `final_verdict`

这个统一入口现在默认直接引用同仓库里的 `vllm-multimodal-evaluator` 媒体目录，不再依赖旧的外部路径名。

## 快速入口

### 一键全量回归

```bash
python scripts/run_full_regression.py \
  --media-root /mnt/sfs_turbo \
  --media-base-url http://127.0.0.1:9000 \
  --auto-start-media-server
```

### 标准复测

```bash
python3 scripts/run_standard_retest.py \
  --host http://127.0.0.1:8000 \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B \
  --media-root /mnt/sfs_turbo \
  --media-base-url http://127.0.0.1:9000
```

### 传输模式一致性回归

```bash
python3 scripts/transport_consistency_check.py \
  --host http://127.0.0.1:8000 \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B/ \
  --image-dir /mnt/sfs_turbo/codes/ascend/vllm-ascend-skill/vllm-multimodal-evaluator/pics/720x1280/jpg \
  --video-path /mnt/sfs_turbo/codes/ascend/vllm-ascend-skill/vllm-multimodal-evaluator/video/720x1280/mp4/shapes.mp4 \
  --media-root /mnt/sfs_turbo \
  --media-base-url http://127.0.0.1:9000 \
  --json
```

### 输出契约自检

```bash
python3 scripts/output_contract_self_check.py \
  --host http://127.0.0.1:8000 \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B \
  --media-root /mnt/sfs_turbo \
  --media-base-url http://127.0.0.1:9000 \
  --json
```

## 默认建议

- 默认同时执行 `base64`、`local_path`、`http`
- 默认同时执行 `L0/L0.5/MME/MMBench`
- 默认同时执行两个 runner 自检
- `enable_thinking=false` 保持不变
- `MMBench` 要保留 `Z` 回退统计
- `MME` 要保留 `unknown` 统计

## 结果解释示例

- `L0` 全通，但 `L0.5/MME/MMBench` 下降
  大概率是模型能力或输出协议问题，不是服务启动问题。
- 只有 `http` 失败
  优先看静态服务、URL 构造、媒体根目录。
- `output_contract` 失败
  先修脚本或输出格式，不要直接判成模型错。
