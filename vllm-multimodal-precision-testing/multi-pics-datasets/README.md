# Multi Pics Dataset

This dataset is the deterministic `1` to `40` image fixture set used by `L0.5`.

It is part of the precision regression flow, not the capability matrix flow.

## What it measures

- single-request multi-image retrieval
- target image index binding
- order sensitivity under higher image counts
- short-answer stability under heavier visual context

## What it does not mean

This dataset is not a direct media-ingestion benchmark.

If a case fails, interpret it like this:

- `wrong` usually points to a model capability limitation
- `unknown` may be model capability, but it may also be output-format or extraction drift
- `timeout`, `request_error`, and `http_xxx` usually point to engineering or serving issues first

## Data layout

```text
cases/<case_id>/
```

Each case contains:

- `question.md`
- `answer.md`
- `answer.json`
- the ordered image files for that case

## Dataset rules

- Case `01` contains exactly `1` image and asks a strict `YES`/`NO` question.
- Cases `02` to `40` contain exactly `N` images and ask for exactly one target image index.
- Shape and color combinations are unique within a case.
- Repeated shapes, if present, always use different colors.

## Rebuild

```bash
python3 multi-pics-datasets/generate_dataset.py
```

## Result files

Standard runs keep:

- `summary.json`
- `summary.csv`
- per-case JSON files

These files should preserve:

- `engineering_errors`
- `model_limitations`
- `output_format_or_protocol_issues`

## How to read failures

- `wrong`
  Model answered, but the answer was wrong.
- `unknown`
  The answer could not be cleanly normalized. Check for explanation-heavy output or extraction instability.
- `timeout` / `request_error`
  First check the serving path and transport path.
