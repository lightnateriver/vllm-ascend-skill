#!/usr/bin/env python3
import argparse
import contextlib
import functools
import json
import shlex
import socketserver
import ssl
import subprocess
import sys
import threading
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.resolve()
DEFAULT_L0_IMAGE_DIR = str((REPO_ROOT / "vllm-multimodal-evaluator" / "pics" / "720x1280" / "jpg").resolve())
DEFAULT_L0_VIDEO_PATH = str((REPO_ROOT / "vllm-multimodal-evaluator" / "video" / "720x1280" / "mp4" / "shapes.mp4").resolve())
DEFAULT_MEDIA_ROOT = "/mnt/sfs_turbo"
DEFAULT_MEDIA_BASE_URL = "http://127.0.0.1:9000"
DEFAULT_MEDIA_MODES = ("base64", "local_path", "http")
DEFAULT_OUTPUT_ROOT = str((SCRIPT_DIR.parent / "regression-runs").resolve())
MME_URL = "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv"
MMBENCH_URL = "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv"


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def is_url_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


@contextlib.contextmanager
def maybe_start_media_server(
    media_base_url: str,
    media_root: str,
    auto_start: bool,
):
    if not auto_start or not media_base_url:
        yield
        return

    parsed = urllib.parse.urlparse(media_base_url)
    host = parsed.hostname or ""
    port = parsed.port or 80
    if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"}:
        yield
        return

    if is_url_reachable(media_base_url, timeout=2.0):
        yield
        return

    root = Path(media_root).expanduser().resolve()
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(root))
    httpd = ReusableThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full multimodal regression stack. By default this now executes "
            "L0, L0.5, MME, and MMBench across base64, local_path, and http."
        )
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_L0_IMAGE_DIR,
    )
    parser.add_argument(
        "--video-path",
        default=DEFAULT_L0_VIDEO_PATH,
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
        "--media-modes",
        nargs="+",
        choices=["base64", "local_path", "http"],
        default=list(DEFAULT_MEDIA_MODES),
        help="Transport modes to execute. Defaults to all three: base64 local_path http.",
    )
    parser.add_argument(
        "--media-mode",
        choices=["base64", "local_path", "http"],
        default=None,
        help="Deprecated single-mode override. If set, the run executes only this one mode.",
    )
    parser.add_argument(
        "--media-root",
        default=DEFAULT_MEDIA_ROOT,
        help="Local media root used for local_path/http modes. For local_path, serve must allow this path.",
    )
    parser.add_argument(
        "--media-base-url",
        default=DEFAULT_MEDIA_BASE_URL,
        help="Base URL for local HTTP media serving, for example http://127.0.0.1:9000 .",
    )
    parser.add_argument(
        "--auto-start-media-server",
        dest="auto_start_media_server",
        action="store_true",
        help="Auto-start a local static HTTP server for localhost/127.0.0.1 media-base-url when needed.",
    )
    parser.add_argument(
        "--no-auto-start-media-server",
        dest="auto_start_media_server",
        action="store_false",
        help="Disable the default local static HTTP server auto-start behavior.",
    )
    parser.set_defaults(auto_start_media_server=True)
    parser.add_argument("--skip-l0", action="store_true")
    parser.add_argument("--skip-l05", action="store_true")
    parser.add_argument("--skip-mme", action="store_true")
    parser.add_argument("--skip-mmbench", action="store_true")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where this regression run stores artifacts and summary files.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run directory name under --output-root.",
    )
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


def build_run_dir(output_root: str, run_name: str | None) -> Path:
    root = Path(output_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return root / (run_name or f"full_regression_{timestamp}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def render_shell_cmd(cmd) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def extract_json_suffix(stdout: str):
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


def run_json_step(name, cmd, step_dir: Path):
    step_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cmd_path = step_dir / "cmd.sh"
    stdout_path = step_dir / "stdout.txt"
    stderr_path = step_dir / "stderr.txt"
    process_meta_path = step_dir / "process.json"
    write_text(cmd_path, render_shell_cmd(cmd) + "\n")
    write_text(stdout_path, proc.stdout)
    write_text(stderr_path, proc.stderr)
    result = {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifacts": {
            "step_dir": str(step_dir),
            "command": str(cmd_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "process": str(process_meta_path),
        },
    }
    result["parsed"] = extract_json_suffix(proc.stdout)
    write_json(
        process_meta_path,
        {
            "name": name,
            "returncode": proc.returncode,
            "command": cmd,
            "parsed_json_available": result["parsed"] is not None,
        },
    )
    return result


def step_passed(step_name, parsed, returncode):
    if step_name == "l0":
        return bool(parsed and parsed.get("summary", {}).get("failed", 1) == 0 and returncode == 0)
    if step_name == "l05":
        return bool(
            parsed
            and parsed.get("wrong", 0) == 0
            and parsed.get("unknown", 0) == 0
            and parsed.get("timeout", 0) == 0
            and returncode == 0
        )
    if step_name == "transport_consistency":
        return bool(parsed and parsed.get("comparison", {}).get("within_tolerance") and returncode == 0)
    if step_name == "output_contract":
        return bool(parsed and parsed.get("summary", {}).get("failed", 1) == 0 and returncode == 0)
    return returncode == 0 and parsed is not None


def build_mode_step(mode, step_name, step_result):
    parsed = step_result["parsed"]
    return {
        "passed": step_passed(step_name, parsed, step_result["returncode"]),
        "returncode": step_result["returncode"],
        "parsed": parsed,
        "stderr": step_result["stderr"].strip(),
        "cmd": step_result["cmd"],
        "media_mode": mode,
        "artifacts": step_result["artifacts"],
    }


def extract_step_precision(summary, step_name, parsed):
    if step_name == "l0":
        summary["precision_summary"]["l0"] = parsed.get("summary", {})
        summary["artifact_paths"]["l0"] = parsed.get("artifacts", {})
        failing_cases = [
            item.get("id")
            for item in parsed.get("results", [])
            if not item.get("pass")
        ]
        engineering_cases = [
            item.get("id")
            for item in parsed.get("results", [])
            if item.get("returncode", 0) != 0
        ]
        model_cases = [case_id for case_id in failing_cases if case_id not in engineering_cases]
        summary["precision_summary"]["l0"]["failing_cases"] = failing_cases
        summary["precision_summary"]["l0"]["failure_class_counts"] = {
            "pipeline_or_serving_issue": len(engineering_cases),
            "model_capability_gap": len(model_cases),
            "output_format_or_extraction_issue": 0,
        }
        summary["engineering_errors"].extend(engineering_cases)
        summary["model_limitations"].extend(model_cases)
        if parsed.get("summary", {}).get("failed", 0):
            summary["final_verdict"]["notes"].append(
                "L0 reported failing smoke cases; these should be triaged before claiming full precision readiness."
            )
    elif step_name == "l05":
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
    elif step_name == "mme":
        summary["precision_summary"]["mme"] = {
            "exact_acc": parsed.get("exact_acc"),
            "unknown": parsed.get("unknown"),
            "perception": parsed.get("scores", {}).get("perception"),
            "reasoning": parsed.get("scores", {}).get("reasoning"),
            "category_scores": parsed.get("scores", {}),
        }
        summary["artifact_paths"]["mme"] = {
            "pred_path": parsed.get("pred_path"),
            "score_path": parsed.get("score_path"),
        }
    elif step_name == "mmbench":
        summary["precision_summary"]["mmbench"] = {
            "overall_acc": parsed.get("exact_acc"),
            "z_fallback": parsed.get("z_fallback"),
            "l2_scores": parsed.get("scores", {}),
            "category_scores": parsed.get("scores", {}),
        }
        summary["artifact_paths"]["mmbench"] = {
            "pred_all_path": parsed.get("pred_all_path"),
            "pred_path": parsed.get("pred_path"),
            "score_path": parsed.get("score_path"),
        }
        summary["engineering_errors"].extend(parsed.get("engineering_error_cases", []))
        summary["model_limitations"].extend(parsed.get("model_limitation_cases", []))
        summary["format_or_protocol_issues"].extend(parsed.get("format_or_protocol_issue_cases", []))
    elif step_name == "transport_consistency":
        summary["precision_summary"]["transport_consistency"] = {
            "within_tolerance": parsed.get("comparison", {}).get("within_tolerance"),
            "accuracy_span": parsed.get("comparison", {}).get("accuracy_span"),
            "totals": parsed.get("comparison", {}).get("totals", []),
            "execution_error_modes": parsed.get("comparison", {}).get("execution_error_modes", {}),
        }
    elif step_name == "output_contract":
        summary["precision_summary"]["output_contract"] = parsed.get("summary", {})


def build_single_mode_summary(mode, step_results):
    summary = {
        "media_mode": mode,
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
        passed = step_passed(step["name"], parsed, step["returncode"])
        summary["steps"][step["name"]] = build_mode_step(mode, step["name"], step)
        summary["overall_pass"] = summary["overall_pass"] and passed
        if parsed:
            extract_step_precision(summary, step["name"], parsed)
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


def summarize_mode_comparison(mode_summaries):
    comparison = {}
    for step_name in ("l0", "l05", "mme", "mmbench"):
        per_mode = {}
        for mode, summary in mode_summaries.items():
            parsed = summary["steps"].get(step_name, {}).get("parsed")
            if not parsed:
                continue
            if step_name == "l0":
                per_mode[mode] = {
                    "passed": parsed.get("summary", {}).get("passed"),
                    "failed": parsed.get("summary", {}).get("failed"),
                    "total": parsed.get("summary", {}).get("total"),
                }
            elif step_name == "l05":
                per_mode[mode] = {
                    "correct": parsed.get("correct"),
                    "wrong": parsed.get("wrong"),
                    "unknown": parsed.get("unknown"),
                    "timeout": parsed.get("timeout"),
                    "accuracy": parsed.get("accuracy"),
                }
            elif step_name == "mme":
                per_mode[mode] = {
                    "exact_acc": parsed.get("exact_acc"),
                    "unknown": parsed.get("unknown"),
                }
            elif step_name == "mmbench":
                per_mode[mode] = {
                    "exact_acc": parsed.get("exact_acc"),
                    "z_fallback": parsed.get("z_fallback"),
                }
        comparison[step_name] = per_mode
    return comparison


def summarize_global_checks(global_checks):
    comparison = {}
    for name, result in global_checks.items():
        parsed = result.get("parsed") or {}
        if name == "transport_consistency":
            comparison[name] = {
                "within_tolerance": parsed.get("comparison", {}).get("within_tolerance"),
                "accuracy_span": parsed.get("comparison", {}).get("accuracy_span"),
                "totals": parsed.get("comparison", {}).get("totals", []),
            }
        elif name == "output_contract":
            comparison[name] = parsed.get("summary", {})
    return comparison


def format_metric(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_metric(item) for item in row) + " |")
    return "\n".join(lines)


def render_mode_comparison_markdown(summary):
    sections = []
    comparison = summary.get("mode_comparison", {})
    step_configs = [
        ("l0", "L0", ["Mode", "Passed", "Failed", "Total"], lambda item: [item.get("passed"), item.get("failed"), item.get("total")]),
        ("l05", "L0.5", ["Mode", "Correct", "Wrong", "Unknown", "Timeout", "Accuracy"], lambda item: [item.get("correct"), item.get("wrong"), item.get("unknown"), item.get("timeout"), item.get("accuracy")]),
        ("mme", "MME", ["Mode", "Exact Acc", "Unknown"], lambda item: [item.get("exact_acc"), item.get("unknown")]),
        ("mmbench", "MMBench", ["Mode", "Overall Acc", "Z Fallback"], lambda item: [item.get("exact_acc"), item.get("z_fallback")]),
    ]
    for step_name, title, headers, row_builder in step_configs:
        per_mode = comparison.get(step_name, {})
        if not per_mode:
            continue
        rows = [[mode, *row_builder(item)] for mode, item in per_mode.items()]
        sections.extend(
            [
                f"## {title}",
                "",
                render_markdown_table(headers, rows),
                "",
            ]
        )
    return "\n".join(sections).rstrip()


def render_global_checks_markdown(summary):
    sections = []
    global_checks = summary.get("global_checks", {})
    if not global_checks:
        return ""
    rows = []
    for name, result in global_checks.items():
        parsed = result.get("parsed") or {}
        if name == "transport_consistency":
            rows.append(
                [
                    name,
                    result.get("passed"),
                    parsed.get("comparison", {}).get("within_tolerance"),
                    parsed.get("comparison", {}).get("accuracy_span"),
                    "",
                ]
            )
        elif name == "output_contract":
            contract_summary = parsed.get("summary", {})
            rows.append(
                [
                    name,
                    result.get("passed"),
                    "",
                    "",
                    f"{contract_summary.get('passed', 0)}/{contract_summary.get('total', 0)} passed",
                ]
            )
    if rows:
        sections.extend(
            [
                "## Global Checks",
                "",
                render_markdown_table(
                    ["Check", "Passed", "Within Tolerance", "Accuracy Span", "Notes"],
                    rows,
                ),
                "",
            ]
        )
    return "\n".join(sections).rstrip()


def render_summary_markdown(summary):
    lines = [
        "# Full Multimodal Regression Summary",
        "",
        render_markdown_table(
            ["Field", "Value"],
            [
                ["run_name", summary.get("run_name")],
                ["artifact_root", summary.get("artifact_root")],
                ["overall_pass", summary.get("overall_pass")],
                ["requested_media_modes", ", ".join(summary.get("requested_media_modes", []))],
            ],
        ),
        "",
        "## Final Verdict",
        "",
    ]
    notes = summary.get("final_verdict", {}).get("notes", [])
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No additional notes.")
    global_checks_md = render_global_checks_markdown(summary)
    if global_checks_md:
        lines.extend(["", global_checks_md])
    lines.extend(["", render_mode_comparison_markdown(summary)])
    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def build_summary(mode_summaries, global_checks, downloads, requested_modes, run_dir: Path):
    mode_pass = all(summary["overall_pass"] for summary in mode_summaries.values()) if mode_summaries else False
    global_pass = all(item.get("passed", False) for item in global_checks.values()) if global_checks else True
    overall_pass = mode_pass and global_pass
    summary = {
        "run_name": run_dir.name,
        "artifact_root": str(run_dir),
        "requested_media_modes": requested_modes,
        "mode_count": len(requested_modes),
        "modes": mode_summaries,
        "mode_comparison": summarize_mode_comparison(mode_summaries),
        "global_checks": global_checks,
        "global_check_summary": summarize_global_checks(global_checks),
        "overall_pass": overall_pass,
        "downloads": downloads,
        "final_verdict": {
            "all_modes_passed": overall_pass,
            "required_modes": requested_modes,
            "notes": [],
        },
    }
    if not overall_pass:
        failed_modes = [mode for mode, item in mode_summaries.items() if not item["overall_pass"]]
        if failed_modes:
            summary["final_verdict"]["notes"].append(
                f"Some transport modes did not fully pass the regression stack: {', '.join(failed_modes)}."
            )
        failed_global = [name for name, item in global_checks.items() if not item.get("passed", False)]
        if failed_global:
            summary["final_verdict"]["notes"].append(
                f"Some regression self-checks failed: {', '.join(failed_global)}."
            )
    return summary


def main():
    args = parse_args()
    downloads = []
    requested_modes = [args.media_mode] if args.media_mode else list(dict.fromkeys(args.media_modes))
    endpoint = f"{args.host.rstrip('/')}/v1/chat/completions"
    run_dir = build_run_dir(args.output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_mme:
        downloads.append(download_if_missing(args.mme_tsv, MME_URL, "mme", not args.no_auto_download))

    if not args.skip_mmbench:
        downloads.append(download_if_missing(args.mmbench_tsv, MMBENCH_URL, "mmbench", not args.no_auto_download))

    with maybe_start_media_server(args.media_base_url, args.media_root, args.auto_start_media_server):
        global_checks = {}
        transport_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "transport_consistency_check.py"),
            "--host",
            args.host,
            "--model",
            args.model,
            "--image-dir",
            args.image_dir,
            "--video-path",
            args.video_path,
            "--media-root",
            args.media_root,
            "--media-base-url",
            args.media_base_url,
            "--json",
        ]
        global_checks["transport_consistency"] = build_mode_step(
            "global",
            "transport_consistency",
            run_json_step("transport_consistency", transport_cmd, run_dir / "global_checks" / "transport_consistency"),
        )
        contract_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "output_contract_self_check.py"),
            "--host",
            args.host,
            "--model",
            args.model,
            "--media-root",
            args.media_root,
            "--media-base-url",
            args.media_base_url,
            "--image-dir",
            args.image_dir,
            "--video-path",
            args.video_path,
            "--l05-dataset-dir",
            args.l05_dataset_dir,
            "--mme-tsv",
            args.mme_tsv,
            "--mmbench-tsv",
            args.mmbench_tsv,
            "--api-key",
            args.api_key,
            "--json",
        ]
        global_checks["output_contract"] = build_mode_step(
            "global",
            "output_contract",
            run_json_step("output_contract", contract_cmd, run_dir / "global_checks" / "output_contract"),
        )

        mode_summaries = {}

        for mode in requested_modes:
            steps = []
            mode_dir = run_dir / "modes" / mode
            if not args.skip_l0:
                cmd = [
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
                    mode,
                    "--json",
                ]
                if args.media_root:
                    cmd.extend(["--media-root", args.media_root])
                if mode == "http" and args.media_base_url:
                    cmd.extend(["--media-base-url", args.media_base_url])
                steps.append(("l0", cmd))

            if not args.skip_l05:
                l05_dir = mode_dir / "l05"
                cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "multi_pics_eval.py"),
                    "--dataset-dir",
                    args.l05_dataset_dir,
                    "--endpoint",
                    endpoint,
                    "--model",
                    args.model,
                    "--media-mode",
                    mode,
                    "--wait-ready",
                    "--output-dir",
                    str(l05_dir),
                    "--json",
                ]
                if args.media_root:
                    cmd.extend(["--media-root", args.media_root])
                if mode == "http" and args.media_base_url:
                    cmd.extend(["--media-base-url", args.media_base_url])
                steps.append(("l05", cmd))

            if not args.skip_mme:
                mme_prefix = mode_dir / "mme" / "mme_qwen35_4b"
                cmd = [
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
                    "--out-prefix",
                    str(mme_prefix),
                    "--media-mode",
                    mode,
                ]
                if args.media_root:
                    cmd.extend(["--media-root", args.media_root])
                if mode == "http" and args.media_base_url:
                    cmd.extend(["--media-base-url", args.media_base_url])
                steps.append(("mme", cmd))

            if not args.skip_mmbench:
                mmbench_prefix = mode_dir / "mmbench" / "mmbench_dev_en_qwen35_4b"
                cmd = [
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
                    "--out-prefix",
                    str(mmbench_prefix),
                    "--media-mode",
                    mode,
                ]
                if args.media_root:
                    cmd.extend(["--media-root", args.media_root])
                if mode == "http" and args.media_base_url:
                    cmd.extend(["--media-base-url", args.media_base_url])
                steps.append(("mmbench", cmd))

            step_results = []
            for name, cmd in steps:
                result = run_json_step(name, cmd, mode_dir / name)
                step_results.append(result)
                if not args.json:
                    print(f"=== {mode} / {name} ===")
                    if result["stdout"].strip():
                        print(result["stdout"].strip())
                    if result["stderr"].strip():
                        print(result["stderr"].strip(), file=sys.stderr)

            mode_summaries[mode] = build_single_mode_summary(mode, step_results)

    summary = build_summary(mode_summaries, global_checks, downloads, requested_modes, run_dir)
    summary_paths = {
        "json": str((run_dir / "summary.json").resolve()),
        "md": str((run_dir / "summary.md").resolve()),
    }
    summary["summary_paths"] = summary_paths
    write_json(run_dir / "summary.json", summary)
    write_text(run_dir / "summary.md", render_summary_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
