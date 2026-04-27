# Multi Pics Dataset

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
