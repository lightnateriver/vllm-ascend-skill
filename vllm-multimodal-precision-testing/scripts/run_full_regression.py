#!/usr/bin/env python3
import argparse
import json
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MME_URL = "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv"
MMBENCH_URL = "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full multimodal regression stack: L0, L0.5 multi-pics, MME, and MMBench."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--image-dir",
        default="/mnt/sfs_turbo/codes/lzp/vllm_multimodal_evaluator/pics/720x1280/jpg",
    )
    parser.add_argument(
        "--video-path",
        default="/mnt/sfs_turbo/codes/lzp/vllm_multimodal_evaluator/video/720x1280/mp4/shapes.mp4",
    )
    parser.add_argument("--mme-tsv", default="/tmp/MME.tsv")
    parser.add_argument("--mmbench-tsv", default="/tmp/MMBench_DEV_EN.tsv")
    parser.add_argument(
        "--l05-dataset-dir",
        default=str((SCRIPT_DIR.parent / "multi-pics-datasets" / "cases").resolve()),
        help="Bundled 1-to-40 multi-pics dataset root.",
    )
    parser.add_argument("--api-key", default="sk-admin")
    parser.add_argument("--max-tokens", type=int, default=8, help="L1 benchmark max completion tokens")
    parser.add_argument("--concurrency", type=int, default=16, help="L1 benchmark concurrency")
    parser.add_argument("--timeout", type=int, default=180, help="L1 benchmark request timeout")
    parser.add_argument(
        "--media-mode",
        choices=["base64", "local_path", "http"],
        default="local_path",
        help="How to send media for L0 and L1 image benchmarks.",
    )
    parser.add_argument(
        "--media-root",
        default="",
        help="Local media root used for local_path/http modes. For local_path, serve must allow this path.",
    )
    parser.add_argument(
        "--media-base-url",
        default="",
        help="Base URL for local HTTP media serving, for example http://127.0.0.1:9000 .",
    )
    parser.add_argument("--skip-l0", action="store_true")
    parser.add_argument("--skip-l05", action="store_true")
    parser.add_argument("--skip-mme", action="store_true")
    parser.add_argument("--skip-mmbench", action="store_true")
    parser.add_argument("--no-auto-download", action="store_true", help="Do not auto-download missing TSV files.")
    parser.add_argument("--json", action="store_true", help="Print only final summary JSON.")
    return parser.parse_args()


def ensure_parent(path_str):
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def download_if_missing(path_str, url, label, auto_download):
    path = Path(path_str).expanduser().resolve()
    if path.exists():
        return {"label": label, "path": str(path), "downloaded": False}
    if not auto_download:
        raise FileNotFoundError(f"{label} TSV not found: {path}")

    ensure_parent(str(path))
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as out:
        out.write(resp.read())
    return {"label": label, "path": str(path), "downloaded": True}


def run_json_step(name, cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        result["parsed"] = json.loads(proc.stdout)
    except Exception:
        result["parsed"] = None
    return result


def build_summary(step_results):
    summary = {
        "steps": {},
        "overall_pass": True,
        "precision_summary": {},
        "engineering_errors": [],
        "model_limitations": [],
        "format_or_protocol_issues": [],
        "artifact_paths": {},
        "final_verdict": {
            "deployment_ready_for_precision": True,
            "precision_status": "unknown",
            "notes": [],
        },
    }
    for step in step_results:
        parsed = step["parsed"]
        if step["name"] == "l0":
            passed = bool(parsed and parsed.get("summary", {}).get("failed", 1) == 0 and step["returncode"] == 0)
        elif step["name"] == "l05":
            passed = bool(
                parsed
                and parsed.get("wrong", 0) == 0
                and parsed.get("unknown", 0) == 0
                and parsed.get("timeout", 0) == 0
                and step["returncode"] == 0
            )
        else:
            passed = step["returncode"] == 0 and parsed is not None

        summary["steps"][step["name"]] = {
            "passed": passed,
            "returncode": step["returncode"],
            "parsed": parsed,
            "stderr": step["stderr"].strip(),
        }
        summary["overall_pass"] = summary["overall_pass"] and passed
        if step["name"] == "l0" and parsed:
            l0_summary = parsed.get("summary", {})
            summary["precision_summary"]["l0"] = l0_summary
            summary["artifact_paths"]["l0"] = parsed.get("artifacts", {})
            if l0_summary.get("failed", 0):
                summary["final_verdict"]["notes"].append(
                    "L0 reported failing smoke cases; these should be triaged before claiming full precision readiness."
                )
        elif step["name"] == "l05" and parsed:
            summary["precision_summary"]["l05"] = {
                "total": parsed.get("total"),
                "correct": parsed.get("correct"),
                "wrong": parsed.get("wrong"),
                "unknown": parsed.get("unknown"),
                "timeout": parsed.get("timeout"),
                "accuracy": parsed.get("accuracy"),
                "failed_cases": parsed.get("failed_cases", []),
                "failure_class_counts": parsed.get("failure_class_counts", {}),
            }
            summary["artifact_paths"]["l05"] = {
                "run_name": parsed.get("run_name"),
                "dataset_dir": parsed.get("dataset_dir"),
            }
            summary["engineering_errors"].extend(parsed.get("engineering_error_cases", []))
            summary["model_limitations"].extend(parsed.get("model_limitation_cases", []))
            summary["format_or_protocol_issues"].extend(
                [
                    item.get("case_id")
                    for item in parsed.get("wrong_cases_detailed", [])
                    if item.get("failure_class") == "output_format_or_extraction_issue"
                ]
            )
        elif step["name"] == "mme" and parsed:
            summary["precision_summary"]["mme"] = {
                "exact_acc": parsed.get("exact_acc"),
                "unknown": parsed.get("unknown"),
                "perception": parsed.get("perception"),
                "reasoning": parsed.get("reasoning"),
                "category_scores": parsed.get("category_scores", {}),
            }
            summary["artifact_paths"]["mme"] = parsed.get("artifacts", {})
        elif step["name"] == "mmbench" and parsed:
            summary["precision_summary"]["mmbench"] = {
                "overall_acc": parsed.get("overall_acc"),
                "z_fallback": parsed.get("z_fallback"),
                "l2_scores": parsed.get("l2_scores", {}),
                "category_scores": parsed.get("category_scores", {}),
            }
            summary["artifact_paths"]["mmbench"] = parsed.get("artifacts", {})
    summary["engineering_errors"] = sorted({str(item) for item in summary["engineering_errors"]})
    summary["model_limitations"] = sorted({str(item) for item in summary["model_limitations"]})
    summary["format_or_protocol_issues"] = sorted({str(item) for item in summary["format_or_protocol_issues"]})
    summary["final_verdict"]["precision_status"] = "pass" if summary["overall_pass"] else "partial_or_fail"
    if summary["engineering_errors"]:
        summary["final_verdict"]["notes"].append(
            "Some failures are currently classified as engineering or serving issues and should be fixed before treating the run as a pure model-quality result."
        )
    if summary["model_limitations"]:
        summary["final_verdict"]["notes"].append(
            "Some failures are currently classified as model capability limitations and should be reported separately from engineering defects."
        )
    return summary


def main():
    args = parse_args()
    steps = []
    downloads = []

    if not args.skip_l0:
        steps.append(
            (
                "l0",
                [
                    sys.executable,
                    str(SCRIPT_DIR / "l0_multimodal_smoke.py"),
                    "--host",
                    args.host,
                    "--model",
                    args.model,
                    "--image-dir",
                    args.image_dir,
                    "--video-path",
                    args.video_path,
                    "--media-mode",
                    args.media_mode,
                    "--json",
                ],
            )
        )
        if args.media_root:
            steps[-1][1].extend(["--media-root", args.media_root])
        if args.media_base_url:
            steps[-1][1].extend(["--media-base-url", args.media_base_url])

    if not args.skip_l05:
        steps.append(
            (
                "l05",
                [
                    sys.executable,
                    str(SCRIPT_DIR / "multi_pics_eval.py"),
                    "--dataset-dir",
                    args.l05_dataset_dir,
                    "--endpoint",
                    endpoint if 'endpoint' in locals() else f"{args.host.rstrip('/')}/v1/chat/completions",
                    "--model",
                    args.model,
                    "--media-mode",
                    args.media_mode,
                    "--wait-ready",
                    "--json",
                ],
            )
        )
        if args.media_root:
            steps[-1][1].extend(["--media-root", args.media_root])
        if args.media_base_url:
            steps[-1][1].extend(["--media-base-url", args.media_base_url])

    endpoint = f"{args.host.rstrip('/')}/v1/chat/completions"

    if not args.skip_mme:
        downloads.append(download_if_missing(args.mme_tsv, MME_URL, "mme", not args.no_auto_download))
        steps.append(
            (
                "mme",
                [
                    sys.executable,
                    str(SCRIPT_DIR / "mme_eval_local.py"),
                    "--tsv",
                    args.mme_tsv,
                    "--endpoint",
                    endpoint,
                    "--model",
                    args.model,
                    "--api-key",
                    args.api_key,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--concurrency",
                    str(args.concurrency),
                    "--timeout",
                    str(args.timeout),
                    "--media-mode",
                    args.media_mode,
                ],
            )
        )
        if args.media_root:
            steps[-1][1].extend(["--media-root", args.media_root])
        if args.media_base_url:
            steps[-1][1].extend(["--media-base-url", args.media_base_url])

    if not args.skip_mmbench:
        downloads.append(download_if_missing(args.mmbench_tsv, MMBENCH_URL, "mmbench", not args.no_auto_download))
        steps.append(
            (
                "mmbench",
                [
                    sys.executable,
                    str(SCRIPT_DIR / "mmbench_eval_local.py"),
                    "--tsv",
                    args.mmbench_tsv,
                    "--endpoint",
                    endpoint,
                    "--model",
                    args.model,
                    "--api-key",
                    args.api_key,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--concurrency",
                    str(args.concurrency),
                    "--timeout",
                    str(args.timeout),
                    "--media-mode",
                    args.media_mode,
                ],
            )
        )
        if args.media_root:
            steps[-1][1].extend(["--media-root", args.media_root])
        if args.media_base_url:
            steps[-1][1].extend(["--media-base-url", args.media_base_url])

    step_results = []
    for name, cmd in steps:
        result = run_json_step(name, cmd)
        step_results.append(result)
        if not args.json:
            print(f"=== {name} ===")
            if result["stdout"].strip():
                print(result["stdout"].strip())
            if result["stderr"].strip():
                print(result["stderr"].strip(), file=sys.stderr)

    summary = build_summary(step_results)
    summary["downloads"] = downloads
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
