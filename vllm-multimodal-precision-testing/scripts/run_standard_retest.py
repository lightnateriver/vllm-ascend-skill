#!/usr/bin/env python3
import argparse
import errno
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from runner_exec_utils import build_python_launch


SCRIPT_DIR = Path(__file__).resolve().parent
PRECISION_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PRECISION_ROOT.parent
CAPABILITY_SCRIPT = REPO_ROOT / "vllm-multimodal-evaluator" / "scripts" / "run_multimodal_capability_tests.py"
PRECISION_SCRIPT = SCRIPT_DIR / "run_full_regression.py"
DEFAULT_HOST = "http://127.0.0.1:8000"
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"
DEFAULT_MEDIA_ROOT = "/mnt/sfs_turbo"
DEFAULT_MEDIA_BASE_URL = "http://127.0.0.1:9000"
DEFAULT_OUTPUT_ROOT = str((PRECISION_ROOT / "retest-runs").resolve())


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard multimodal retest flow: capability first, then the "
            "three-mode full precision regression (L0/L0.5/MME/MMBench)."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Service root, for example http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--media-root", default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    parser.add_argument("--api-key", default="sk-admin")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None, help="Optional explicit run directory name.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunked-prefill", type=str2bool, default=True)
    parser.add_argument("--async-scheduling", type=str2bool, default=True)
    parser.add_argument("--prefix-caching", type=str2bool, default=True)
    parser.add_argument("--function-calling", type=str2bool, default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--skip-capability", action="store_true")
    parser.add_argument("--skip-precision", action="store_true")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument(
        "--no-auto-start-media-server",
        dest="auto_start_media_server",
        action="store_false",
        help="Disable the shared localhost static media server auto-start behavior.",
    )
    parser.set_defaults(auto_start_media_server=True)
    parser.add_argument("--json", action="store_true", help="Print only the final machine-readable summary JSON.")
    return parser.parse_args()


def build_run_dir(output_root: str, run_name: str | None) -> Path:
    root = Path(output_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return root / (run_name or f"standard_retest_{timestamp}")


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


def run_and_capture(name: str, launch, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = launch["cmd"]
    env = launch.get("env")
    launch_meta = launch.get("meta", {})
    launch_error = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except OSError as exc:
        proc = None
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        launch_error = stderr
        returncode = 127 if getattr(exc, "errno", None) == errno.ENOENT or isinstance(exc, FileNotFoundError) else 126
    cmd_path = work_dir / "cmd.sh"
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"
    write_text(cmd_path, render_shell_cmd(cmd) + "\n")
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    result = {
        "name": name,
        "cmd": cmd,
        "launch": launch_meta,
        "launch_error": launch_error,
        "returncode": returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(cmd_path),
    }
    if stdout:
        result["parsed_json"] = extract_json_suffix(stdout)
    return result


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_capability(report_json):
    if not report_json:
        return {}
    summary = report_json.get("summary", report_json)
    preflight = report_json.get("preflight", {})
    return {
        "counts_by_status": summary.get("counts_by_status", {}),
        "counts_by_test_type": summary.get("counts_by_test_type", {}),
        "included_test_items": summary.get("included_test_items", []),
        "engineering_errors": summary.get("engineering_errors", []),
        "model_limitations": summary.get("model_limitations", []),
        "non_pass_case_count": len(summary.get("non_pass_cases", [])),
        "runtime_warnings": preflight.get("runtime_warnings", []),
    }


def summarize_precision(report_json):
    if not report_json:
        return {}
    return {
        "overall_pass": report_json.get("overall_pass"),
        "requested_media_modes": report_json.get("requested_media_modes", []),
        "mode_comparison": report_json.get("mode_comparison", {}),
        "global_check_summary": report_json.get("global_check_summary", {}),
        "final_verdict": report_json.get("final_verdict", {}),
    }


def format_metric(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_metric(item) for item in row) + " |")
    return "\n".join(lines)


def render_mode_comparison_tables(mode_comparison):
    sections = []
    step_configs = [
        ("l0", "L0", ["Mode", "Passed", "Failed", "Total"], lambda item: [item.get("passed"), item.get("failed"), item.get("total")]),
        ("l05", "L0.5", ["Mode", "Correct", "Wrong", "Unknown", "Timeout", "Accuracy"], lambda item: [item.get("correct"), item.get("wrong"), item.get("unknown"), item.get("timeout"), item.get("accuracy")]),
        ("mme", "MME", ["Mode", "Exact Acc", "Unknown"], lambda item: [item.get("exact_acc"), item.get("unknown")]),
        ("mmbench", "MMBench", ["Mode", "Overall Acc", "Z Fallback"], lambda item: [item.get("exact_acc"), item.get("z_fallback")]),
    ]
    for step_name, title, headers, row_builder in step_configs:
        per_mode = mode_comparison.get(step_name, {})
        if not per_mode:
            continue
        rows = [[mode, *row_builder(item)] for mode, item in per_mode.items()]
        sections.extend([f"## {title}", "", render_table(headers, rows), ""])
    return "\n".join(sections).rstrip()


def render_summary_markdown(summary):
    lines = [
        "# Standard Multimodal Retest Summary",
        "",
        render_table(
            ["Field", "Value"],
            [
                ["run_name", summary.get("run_name")],
                ["run_dir", summary.get("run_dir")],
                ["host", summary.get("host")],
                ["model", summary.get("model")],
                ["media_base_url", summary.get("media_base_url")],
                ["capability_returncode", summary.get("capability", {}).get("returncode")],
                ["precision_returncode", summary.get("precision", {}).get("returncode")],
            ],
        ),
        "",
        "## Artifacts",
        "",
        render_table(
            ["Artifact", "Path"],
            [
                ["top_summary_json", summary.get("summary_paths", {}).get("json")],
                ["top_summary_md", summary.get("summary_paths", {}).get("md")],
                ["capability_report_json", summary.get("capability", {}).get("report_json")],
                ["capability_report_md", summary.get("capability", {}).get("report_md")],
                ["precision_summary_json", summary.get("precision", {}).get("summary_json")],
                ["precision_summary_md", summary.get("precision", {}).get("summary_md")],
            ],
        ),
        "",
    ]
    capability_summary = summary.get("capability", {}).get("summary", {})
    if capability_summary:
        lines.extend(
            [
                "## Capability Snapshot",
                "",
                render_table(
                    ["Metric", "Value"],
                    [
                        ["non_pass_case_count", capability_summary.get("non_pass_case_count")],
                        ["engineering_errors", len(capability_summary.get("engineering_errors", []))],
                        ["model_limitations", len(capability_summary.get("model_limitations", []))],
                    ],
                ),
                "",
            ]
        )
        capability_warnings = capability_summary.get("runtime_warnings", [])
        if capability_warnings:
            lines.extend(
                [
                    "### Capability Runtime Warnings",
                    "",
                ]
            )
            for warning in capability_warnings:
                lines.append(f"- {warning}")
            lines.append("")
        included_items = capability_summary.get("included_test_items", [])
        if included_items:
            lines.extend(
                [
                    "### Capability Test Items",
                    "",
                    render_table(
                        ["Suite", "Category", "Type", "Transports", "Total", "PASS", "FAIL", "BLOCKED"],
                        [
                            [
                                item.get("suite_name"),
                                item.get("category"),
                                item.get("test_type"),
                                ", ".join(item.get("transport_modes", [])),
                                item.get("total_cases"),
                                item.get("PASS", 0),
                                item.get("FAIL", 0),
                                item.get("BLOCKED", 0),
                            ]
                            for item in included_items
                        ],
                    ),
                    "",
                ]
            )
    precision_summary = summary.get("precision", {}).get("summary", {})
    if precision_summary:
        lines.extend(
            [
                "## Precision Verdict",
                "",
                render_table(
                    ["Field", "Value"],
                    [
                        ["overall_pass", precision_summary.get("overall_pass")],
                        ["requested_media_modes", ", ".join(precision_summary.get("requested_media_modes", []))],
                    ],
                ),
                "",
                "### Precision Global Checks",
                "",
                render_table(
                    ["Check", "Value"],
                    [[key, json.dumps(value, ensure_ascii=False)] for key, value in precision_summary.get("global_check_summary", {}).items()],
                ) if precision_summary.get("global_check_summary") else "No global checks recorded.",
                "",
                render_mode_comparison_tables(precision_summary.get("mode_comparison", {})),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    run_dir = build_run_dir(args.output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "host": args.host,
        "model": args.model,
        "media_root": args.media_root,
        "media_base_url": args.media_base_url,
    }

    capability_dir = run_dir / "capability"
    capability_report_json = capability_dir / "qwen35_multimodal_capability_report.json"
    capability_report_md = capability_dir / "qwen35_multimodal_capability_report.md"
    if args.skip_capability:
        summary["capability"] = {"skipped": True, "returncode": None}
    else:
        capability_launch = build_python_launch(
            CAPABILITY_SCRIPT,
            "--base-url",
            f"{args.host.rstrip('/')}/v1",
            "--model",
            args.model,
            "--results-dir",
            str(capability_dir),
            "--dtype",
            args.dtype,
            "--chunked-prefill",
            str(args.chunked_prefill),
            "--async-scheduling",
            str(args.async_scheduling),
            "--prefix-caching",
            str(args.prefix_caching),
            "--function-calling",
            str(args.function_calling),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--enforce-eager",
            str(args.enforce_eager),
            "--media-base-url",
            args.media_base_url,
            "--json",
        )
        capability_launch["cmd"].append("--auto-start-media-server" if args.auto_start_media_server else "--no-auto-start-media-server")
        capability_result = run_and_capture("capability", capability_launch, capability_dir / "_runner")
        capability_payload = load_json_if_exists(capability_report_json) or capability_result.get("parsed_json")
        capability_result["report_json"] = str(capability_report_json)
        capability_result["report_md"] = str(capability_report_md)
        capability_result["summary"] = summarize_capability(capability_payload)
        summary["capability"] = capability_result

    precision_root = run_dir / "precision"
    precision_run_name = "precision_full"
    precision_run_dir = precision_root / precision_run_name
    precision_summary_json = precision_run_dir / "summary.json"
    precision_summary_md = precision_run_dir / "summary.md"
    if args.skip_precision:
        summary["precision"] = {"skipped": True, "returncode": None}
    else:
        precision_launch = build_python_launch(
            PRECISION_SCRIPT,
            "--host",
            args.host,
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
            "--media-root",
            args.media_root,
            "--media-base-url",
            args.media_base_url,
            "--output-root",
            str(precision_root),
            "--run-name",
            precision_run_name,
            "--json",
        )
        if args.no_auto_download:
            precision_launch["cmd"].append("--no-auto-download")
        precision_launch["cmd"].append("--auto-start-media-server" if args.auto_start_media_server else "--no-auto-start-media-server")
        precision_result = run_and_capture("precision", precision_launch, precision_root / "_runner")
        precision_payload = load_json_if_exists(precision_summary_json) or precision_result.get("parsed_json")
        precision_result["summary_json"] = str(precision_summary_json)
        precision_result["summary_md"] = str(precision_summary_md)
        precision_result["summary"] = summarize_precision(precision_payload)
        summary["precision"] = precision_result

    summary_paths = {
        "json": str((run_dir / "retest_summary.json").resolve()),
        "md": str((run_dir / "retest_summary.md").resolve()),
    }
    summary["summary_paths"] = summary_paths
    write_json(run_dir / "retest_summary.json", summary)
    write_text(run_dir / "retest_summary.md", render_summary_markdown(summary))

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"standard retest artifacts: {run_dir}")
        print(f"summary json: {summary_paths['json']}")
        print(f"summary md: {summary_paths['md']}")

    capability_ok = summary.get("capability", {}).get("skipped") or summary.get("capability", {}).get("returncode") == 0
    precision_ok = summary.get("precision", {}).get("skipped") or summary.get("precision", {}).get("returncode") == 0
    return 0 if capability_ok and precision_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
