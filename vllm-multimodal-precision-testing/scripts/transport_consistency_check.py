#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import errno
import functools
import json
import subprocess
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from runner_exec_utils import build_python_cmd

SCRIPT_DIR = Path(__file__).resolve().parent


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


@dataclass
class ModeSummary:
    mode: str
    passed: int
    failed: int
    total: int
    execution_error_count: int

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight transport consistency regression across base64, local_path, and http."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B/")
    parser.add_argument(
        "--image-dir",
        default="/mnt/sfs_turbo/codes/ascend/vllm-ascend-skill/vllm-multimodal-evaluator/pics/720x1280/jpg",
    )
    parser.add_argument(
        "--video-path",
        default="/mnt/sfs_turbo/codes/ascend/vllm-ascend-skill/vllm-multimodal-evaluator/video/720x1280/mp4/shapes.mp4",
    )
    parser.add_argument("--media-root", default="/mnt/sfs_turbo")
    parser.add_argument("--media-base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def is_url_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


def resolve_media_probe_path(image_dir: str, video_path: str) -> Path | None:
    candidates = [
        Path(image_dir).expanduser().resolve() / "circle.jpg",
        Path(image_dir).expanduser().resolve() / "rectangle.jpg",
        Path(video_path).expanduser().resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_media_probe_url(media_base_url: str, media_root: str, probe_path: Path | None) -> str | None:
    if probe_path is None:
        return None
    resolved_root = Path(media_root).expanduser().resolve()
    try:
        relative = probe_path.resolve().relative_to(resolved_root)
    except ValueError:
        return None
    quoted = "/".join(urllib.parse.quote(part) for part in relative.parts)
    return f"{media_base_url.rstrip('/')}/{quoted}"


@contextlib.contextmanager
def maybe_start_media_server(media_base_url: str, media_root: str, probe_path: Path | None = None):
    parsed = urllib.parse.urlparse(media_base_url)
    host = parsed.hostname or ""
    port = parsed.port or 80
    if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"}:
        yield
        return

    probe_url = build_media_probe_url(media_base_url, media_root, probe_path)
    if is_url_reachable(media_base_url, timeout=2.0):
        if probe_url is None or is_url_reachable(probe_url, timeout=2.0):
            yield
            return
        # HTTP child runs can self-host a correct local media server and pick an
        # alternate localhost port when a conflicting shared server is present.
        yield
        return

    root = Path(media_root).expanduser().resolve()
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(root))
    try:
        httpd = ReusableThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            yield
            return
        raise
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()


def run_l0(mode: str, args: argparse.Namespace) -> dict:
    cmd = build_python_cmd(
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
        mode,
        "--json",
    )
    if mode != "base64":
        cmd.extend(["--media-root", args.media_root])
    if mode == "http":
        cmd.extend(["--media-base-url", args.media_base_url])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        return {
            "mode": mode,
            "returncode": proc.returncode,
            "summary": {"passed": 0, "failed": 0, "total": 0},
            "stderr": (proc.stderr or proc.stdout).strip(),
            "execution_error_count": 1,
            "execution_error_cases": [f"{mode}:runner_parse_failure"],
        }
    summary = parsed["summary"]
    execution_error_cases = [
        item["id"]
        for item in parsed.get("results", [])
        if item.get("returncode", 0) != 0
    ]
    return {
        "mode": mode,
        "returncode": proc.returncode,
        "summary": summary,
        "stderr": proc.stderr.strip(),
        "execution_error_count": len(execution_error_cases),
        "execution_error_cases": execution_error_cases,
    }


def compare_modes(results: list[dict], tolerance: float) -> dict:
    summaries = [
        ModeSummary(
            mode=item["mode"],
            passed=item["summary"]["passed"],
            failed=item["summary"]["failed"],
            total=item["summary"]["total"],
            execution_error_count=item["execution_error_count"],
        )
        for item in results
    ]
    totals = sorted({item.total for item in summaries})
    accuracies = {item.mode: round(item.accuracy, 4) for item in summaries}
    accuracy_span = max(item.accuracy for item in summaries) - min(item.accuracy for item in summaries)
    execution_error_modes = {
        item.mode: item.execution_error_count
        for item in summaries
        if item.execution_error_count > 0
    }
    return {
        "totals": totals,
        "accuracies": accuracies,
        "accuracy_span": round(accuracy_span, 6),
        "execution_error_modes": execution_error_modes,
        "within_tolerance": (
            accuracy_span <= tolerance
            and len(totals) == 1
            and not execution_error_modes
        ),
    }


def main() -> int:
    args = parse_args()
    results = [run_l0(mode, args) for mode in ("base64", "local_path", "http")]
    comparison = compare_modes(results, args.tolerance)
    payload = {
        "results": results,
        "comparison": comparison,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in results:
            summary = item["summary"]
            print(
                f"{item['mode']}: "
                f"passed={summary['passed']} failed={summary['failed']} total={summary['total']} "
                f"execution_errors={item['execution_error_count']}"
            )
        print(
            "comparison: "
            f"totals={comparison['totals']} "
            f"accuracies={comparison['accuracies']} "
            f"accuracy_span={comparison['accuracy_span']} "
            f"execution_error_modes={comparison['execution_error_modes']} "
            f"within_tolerance={comparison['within_tolerance']}"
        )
    return 0 if comparison["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
