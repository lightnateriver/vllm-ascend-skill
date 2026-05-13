# Multi Pics Dataset

这个数据集服务于 `L0.5` 多图精度测试，不是 capability 支持矩阵测试。

建议按下面口径解读结果：

- `wrong` 更接近模型能力边界，例如多图检索、顺序敏感或短答案不稳定。
- `unknown` 不直接等于视觉失败，也可能是输出格式没有收敛到要求答案。
- `timeout`、`request_error`、`http_xxx` 更接近服务或链路问题，应优先按工程问题排查。

建议统一保留：

- `summary.json`
- `summary.csv`
- 每个 case 的逐题 JSON

并在最终汇总时明确区分：

- `engineering_errors`
- `model_limitations`
- `output_format_or_protocol_issues`

This directory contains a deterministic `1` to `40` image precision dataset for multimodal regression checks.

Structure:

- `generate_dataset.py`: rebuilds the full dataset with a fixed seed
- `cases/`: generated case directories from `01` to `40`

Dataset rules:

- Case `01` contains exactly `1` image.
- Cases `02` to `40` contain exactly `N` images for case `N`.
- Case `01` asks a `YES` or `NO` question.
- Cases `02` to `40` ask for one target image index and require exactly one number.
- Shape and color combinations are unique inside each case.
- If a shape repeats in one case, its color is different.

Each case directory includes:

- `question.md`
- `answer.md`
- `answer.json`
- ordered `*.jpg` images

Use:

```bash
python3 multi-pics-datasets/generate_dataset.py
```
