#!/usr/bin/env python3
import argparse
import json
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_IMAGE_DIR = SKILL_DIR / "assets" / "l0" / "pics" / "720x1280" / "jpg"
DEFAULT_VIDEO_PATH = SKILL_DIR / "assets" / "l0" / "video" / "720x1280" / "mp4" / "shapes.mp4"
MME_URL = "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv"
MMBENCH_URL = "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full multimodal regression stack: L0 smoke, MME, and MMBench."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
    )
    parser.add_argument(
        "--video-path",
        default=str(DEFAULT_VIDEO_PATH),
    )
    parser.add_argument("--mme-tsv", default="/tmp/MME.tsv")
    parser.add_argument("--mmbench-tsv", default="/tmp/MMBench_DEV_EN.tsv")
    parser.add_argument("--api-key", default="sk-admin")
    parser.add_argument("--max-tokens", type=int, default=8, help="L1 benchmark max completion tokens")
    parser.add_argument("--concurrency", type=int, default=8, help="L1 benchmark concurrency")
    parser.add_argument("--timeout", type=int, default=180, help="L1 benchmark request timeout")
    parser.add_argument("--skip-l0", action="store_true")
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
    summary = {"steps": {}, "overall_pass": True}
    for step in step_results:
        parsed = step["parsed"]
        if step["name"] == "l0":
            passed = bool(parsed and parsed.get("summary", {}).get("failed", 1) == 0 and step["returncode"] == 0)
        else:
            passed = step["returncode"] == 0 and parsed is not None

        summary["steps"][step["name"]] = {
            "passed": passed,
            "returncode": step["returncode"],
            "parsed": parsed,
            "stderr": step["stderr"].strip(),
        }
        summary["overall_pass"] = summary["overall_pass"] and passed
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
                    "--json",
                ],
            )
        )

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
                ],
            )
        )

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
                ],
            )
        )

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
