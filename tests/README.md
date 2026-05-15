# Skill Regression Tests

This directory holds lightweight regression tests for the skill code itself.

Current coverage:

- `test_output_contracts.py`
  Verifies machine-readable output expectations for script-facing JSON contracts.
- `test_transport_consistency.py`
  Verifies the comparison logic used for `base64`, `local_path`, and `http` transport-mode consistency checks.

These tests do not replace the full online model evaluations.
They are meant to catch:

- broken JSON output expectations
- accidental summary-schema drift
- transport-mode comparison logic regressions
- false-positive transport consistency passes caused by request or execution errors

Run the lightweight self-check suite with:

```bash
python3 tests/run_self_checks.py
```

Run the online transport consistency regression with:

```bash
python3 vllm-multimodal-precision-testing/scripts/transport_consistency_check.py \
  --host http://127.0.0.1:8000 \
  --model /mnt/sfs_turbo/models/Qwen/Qwen3.5-4B/ \
  --image-dir vllm-multimodal-evaluator/pics/720x1280/jpg \
  --video-path vllm-multimodal-evaluator/video/720x1280/mp4/shapes.mp4 \
  --media-root /mnt/sfs_turbo \
  --media-base-url http://127.0.0.1:9000 \
  --json
```
