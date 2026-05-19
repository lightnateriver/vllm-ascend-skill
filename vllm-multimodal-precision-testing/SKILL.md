---
name: vllm-multimodal-precision-testing
description: Run layered multimodal precision regression for local vLLM or vllm-ascend services. Standard runs execute L0, L0.5, MME, and MMBench across base64, local_path, and http by default, plus transport consistency and output contract self-checks.
---

# vLLM Multimodal Precision Testing

## What this skill is for

Use this skill when the service is already up and you need to answer:

> Did the model or the multimodal serving path regress in precision?

This is the regression layer after basic capability is already known to work.

## What it covers

This skill currently covers:

- `L0`
- `L0.5`
- `MME`
- `MMBench_DEV_EN`
- transport consistency regression
- output contract self-check

## Default behavior

Standard regression defaults to:

- all three transport modes:
  - `base64`
  - `local_path`
  - `http`
- all major precision suites:
  - `L0`
  - `L0.5`
  - `MME`
  - `MMBench`
- both runner-level self-checks:
  - `transport_consistency_check.py`
  - `output_contract_self_check.py`

## Test purposes

### L0

Catch obvious regressions in:

- single-image recognition
- multi-image order handling
- short constrained answer following
- basic video understanding

### L0.5

Catch regressions that only appear when one request carries many images:

- target index binding
- order sensitivity
- 1 to 40 image scaling
- short answer stability under heavier visual context

### MME

Track broader yes/no perception and reasoning coverage, especially:

- OCR
- existence
- count
- position
- translation
- calculation

### MMBench

Track broader MCQ perception and reasoning coverage, including:

- localization
- spatial relationship
- OCR
- structure reading
- future prediction

### transport_consistency

Check whether `base64`, `local_path`, and `http` behave consistently on the same L0 sample set.

This is a runner-level regression guard, not a benchmark.

### output_contract

Check that the JSON contracts of the precision scripts remain machine-readable and stable.

This prevents automation from breaking when a script starts mixing logs into its JSON output.

## Transport modes

The three transport modes have different roles:

| Mode | What it means | Main use |
| --- | --- | --- |
| `base64` | `data:` URLs | Default compatibility path |
| `local_path` | `file://` URLs | Validate local file access path |
| `http` | local static media URLs | Validate URL-based media access |

`local_path` is the user-facing name. Internally, it is still implemented with a `file://` URL.

## Failure attribution

Do not collapse every failure into “model bad”.

Use these three categories:

- `Engineering Error`
  Service startup, HTTP request failure, media path failure, timeout, script error, static server issue.
- `Model Capability Limitation`
  Media was read successfully, but the answer was actually wrong.
- `Output Format / Protocol Issue`
  The model answered, but not in the required short form, yes/no form, or option-letter form.

### Practical rule

- `L0` failure: first check transport and serving
- `L0.5 wrong_answer`: usually model capability
- `MME unknown`: may be output-format or extraction instability, not necessarily model incapability
- `MMBench Z`: usually output-format or protocol drift
- `transport_consistency` failure: usually media-serving or URL construction

## Report expectations

Reports should show:

- what was tested
- which transports were included
- which suites were enabled by default
- the outcome of each suite
- pass / fail details
- failure attribution

Important report fields:

- `requested_media_modes`
- `mode_comparison`
- `global_checks`
- `global_check_summary`
- `precision_summary`
- `engineering_errors`
- `model_limitations`
- `format_or_protocol_issues`
- `final_verdict`

## Recommended flow

1. Confirm the service is up.
2. Run `vllm-multimodal-evaluator` if the transport path or media path is uncertain.
3. Run `run_full_regression.py` for default three-mode precision.
4. If needed, run `run_standard_retest.py` to combine capability and precision in one artifact tree.

## Default command

```bash
python scripts/run_full_regression.py \
  --media-root /mnt/sfs_turbo \
  --media-base-url http://127.0.0.1:9000 \
  --auto-start-media-server
```

## Summary rule

When summarizing results for a user, always say:

- what suite passed or failed
- what transport mode passed or failed
- whether the issue is engineering, protocol, or model capability
- whether `unknown` or `Z` is a real model error or an extraction issue
