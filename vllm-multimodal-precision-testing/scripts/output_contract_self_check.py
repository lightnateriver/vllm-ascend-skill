#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from runner_exec_utils import build_python_cmd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "http://127.0.0.1:8000"
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"
DEFAULT_MEDIA_ROOT = "/mnt/sfs_turbo"
DEFAULT_MEDIA_BASE_URL = "http://127.0.0.1:9000"
DEFAULT_L0_IMAGE_DIR = str((SCRIPT_DIR.parent.parent / "vllm-multimodal-evaluator" / "pics" / "720x1280" / "jpg").resolve())
DEFAULT_L0_VIDEO_PATH = str((SCRIPT_DIR.parent.parent / "vllm-multimodal-evaluator" / "video" / "720x1280" / "mp4" / "shapes.mp4").resolve())
DEFAULT_L05_DATASET_DIR = str((SCRIPT_DIR.parent / "multi-pics-datasets" / "cases").resolve())
DEFAULT_OUTPUT_ROOT = Path("/tmp/precision_contract_self_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight self-check that validates the JSON output contracts of precision scripts."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--media-root", default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    parser.add_argument("--image-dir", default=DEFAULT_L0_IMAGE_DIR)
    parser.add_argument("--video-path", default=DEFAULT_L0_VIDEO_PATH)
    parser.add_argument("--l05-dataset-dir", default=DEFAULT_L05_DATASET_DIR)
    parser.add_argument("--mme-tsv", default="/tmp/MME.tsv")
    parser.add_argument("--mmbench-tsv", default="/tmp/MMBench_DEV_EN.tsv")
    parser.add_argument("--api-key", default="sk-admin")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def required_keys_present(payload: dict[str, Any], keys: list[str]) -> list[str]:
    return [key for key in keys if key not in payload]


def extract_json(stdout: str) -> dict[str, Any] | None:
    lines = stdout.splitlines()
    for idx, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        candidate = "\n".join(lines[idx:])
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def run_and_check(name: str, cmd: list[str], required_keys: list[str], nested_keys: list[tuple[str, list[str]]]) -> dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    parsed = extract_json(proc.stdout)
    issues: list[str] = []
    if parsed is None:
        issues.append("missing_json_payload")
    else:
        missing = required_keys_present(parsed, required_keys)
        issues.extend([f"missing_key:{key}" for key in missing])
        for parent, children in nested_keys:
            parent_obj = parsed.get(parent)
            if not isinstance(parent_obj, dict):
                issues.append(f"missing_object:{parent}")
                continue
            issues.extend([f"missing_key:{parent}.{key}" for key in required_keys_present(parent_obj, children)])
    return {
        "name": name,
        "returncode": proc.returncode,
        "passed": not issues and parsed is not None,
        "issues": issues,
        "stdout_excerpt": proc.stdout[:500],
        "stderr_excerpt": proc.stderr[:500],
    }


def main() -> int:
    args = parse_args()
    endpoint = f"{args.host.rstrip('/')}/v1/chat/completions"
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checks = [
        run_and_check(
            "l0",
            build_python_cmd(
                SCRIPT_DIR / "l0_multimodal_smoke.py",
                "--host",
                args.host,
                "--model",
                args.model,
                "--image-dir",
                args.image_dir,
                "--video-path",
                args.video_path,
                "--media-mode",
                "base64",
                "--json",
            ),
            required_keys=["summary", "results"],
            nested_keys=[("summary", ["passed", "failed", "total", "media_mode"])],
        ),
        run_and_check(
            "l05",
            build_python_cmd(
                SCRIPT_DIR / "multi_pics_eval.py",
                "--dataset-dir",
                args.l05_dataset_dir,
                "--case-range",
                "01:02",
                "--endpoint",
                endpoint,
                "--model",
                args.model,
                "--media-mode",
                "base64",
                "--json",
            ),
            required_keys=["run_name", "total", "correct", "wrong", "unknown", "timeout", "accuracy", "failure_class_counts"],
            nested_keys=[],
        ),
    ]

    if Path(args.mme_tsv).exists():
        checks.append(
            run_and_check(
                "mme",
                build_python_cmd(
                    SCRIPT_DIR / "mme_eval_local.py",
                    "--tsv",
                    args.mme_tsv,
                    "--limit",
                    "4",
                    "--endpoint",
                    endpoint,
                    "--model",
                    args.model,
                    "--api-key",
                    args.api_key,
                    "--out-prefix",
                    str(DEFAULT_OUTPUT_ROOT / "mme_contract"),
                    "--media-mode",
                    "base64",
                ),
                required_keys=["rows", "exact_acc", "unknown", "pred_path", "score_path", "scores"],
                nested_keys=[],
            )
        )

    if Path(args.mmbench_tsv).exists():
        checks.append(
            run_and_check(
                "mmbench",
                build_python_cmd(
                    SCRIPT_DIR / "mmbench_eval_local.py",
                    "--tsv",
                    args.mmbench_tsv,
                    "--limit",
                    "4",
                    "--endpoint",
                    endpoint,
                    "--model",
                    args.model,
                    "--api-key",
                    args.api_key,
                    "--out-prefix",
                    str(DEFAULT_OUTPUT_ROOT / "mmbench_contract"),
                    "--media-mode",
                    "base64",
                ),
                required_keys=["rows_all", "rows_scored", "exact_acc", "z_fallback", "pred_all_path", "pred_path", "score_path", "scores"],
                nested_keys=[],
            )
        )

    payload = {
        "checks": checks,
        "summary": {
            "passed": sum(1 for item in checks if item["passed"]),
            "failed": sum(1 for item in checks if not item["passed"]),
            "total": len(checks),
        },
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"{status} {item['name']} issues={','.join(item['issues']) or 'none'}")
        print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
