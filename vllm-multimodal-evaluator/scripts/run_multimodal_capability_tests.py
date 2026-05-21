from __future__ import annotations

import argparse
import contextlib
import errno
import functools
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from capability_case_profiles import (
    LARGE_IMAGE_RESOLUTIONS,
    STANDARD_TRANSPORT_MODES,
    TRANSPORT_LABELS,
    TestCase,
    assert_case_files,
    build_capability_cases,
)
from runner_exec_utils import build_python_cmd, build_python_launch


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"
DEFAULT_MEDIA_BASE_URL = "http://127.0.0.1:9000"
FC_SUITE_NAME = "phase2_function_calling_standard"
DEFAULT_FC_TEST_FILE = Path(__file__).resolve().with_name("function_calling_test.json")
SCRIPT_DIR = Path(__file__).resolve().parent

SERVICE_CONFIG_DEFAULTS = {
    "chunked_prefill": True,
    "async_scheduling": True,
    "prefix_caching": True,
    "function_calling": True,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.7,
    "enforce_eager": True,
}


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def ensure_local_media(
    project_root: Path,
    include_large_image_smoke: bool,
    large_image_resolutions: list[str],
) -> None:
    pics_root = project_root / "pics"
    video_root = project_root / "video"
    standard_image_probes = [
        pics_root / "720x1280" / "tiff" / "triangle.tiff",
        pics_root / "1920x1080" / "jpg" / "circle.jpg",
        pics_root / "720x1280" / "jpg" / "cube.jpg",
    ]
    if any(not path.exists() for path in standard_image_probes):
        subprocess.run(
            build_python_cmd(SCRIPT_DIR / "generate_shape_dataset.py", "--resolution-profile", "standard"),
            cwd=project_root,
            check=True,
        )
    standard_video_probes = [
        video_root / "720x1280" / "avi" / "shapes.avi",
        video_root / "1080x1920" / "mp4" / "shapes.mp4",
        video_root / "720x1280" / "mp4" / "square.mp4",
    ]
    if any(not path.exists() for path in standard_video_probes):
        subprocess.run(
            build_python_cmd(SCRIPT_DIR / "generate_shape_videos.py", "--resolution-profile", "standard"),
            cwd=project_root,
            check=True,
        )
    if include_large_image_smoke:
        large_image_probes = [pics_root / resolution / "jpg" / "square.jpg" for resolution in large_image_resolutions]
        if any(not path.exists() for path in large_image_probes):
            cmd = [
                *build_python_cmd(SCRIPT_DIR / "generate_shape_dataset.py"),
                "--resolution-profile",
                "large",
                "--formats",
                "jpg",
            ]
            if large_image_resolutions:
                cmd.extend(["--resolutions", *large_image_resolutions])
            subprocess.run(cmd, cwd=project_root, check=True)


def curl_json(url: str, payload: dict[str, Any] | None, timeout: float, method: str) -> tuple[int, dict[str, Any] | None, str]:
    marker = "__CODEX_HTTP_STATUS__"
    timeout_str = f"{max(timeout, 1.0):g}"
    cmd = [
        "curl",
        "-sS",
        "--connect-timeout",
        timeout_str,
        "--max-time",
        timeout_str,
        "-H",
        "Content-Type: application/json",
    ]
    if method != "GET":
        cmd.extend(["-X", method])
    if payload is not None:
        cmd.extend(["--data-binary", "@-"])
    cmd.extend(["-w", f"\n{marker}%{{http_code}}", url])

    input_bytes = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()

    status = 0
    body = stdout
    if marker in stdout:
        body, status_text = stdout.rsplit(marker, 1)
        try:
            status = int(status_text.strip() or "0")
        except ValueError:
            status = 0

    if proc.returncode != 0 and status == 0 and not body:
        return 0, None, stderr or f"curl return code {proc.returncode}"

    parsed = None
    if body.strip():
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None

    return status, parsed, body or stderr


def resolve_model_id(requested_model: str, available_ids: set[str]) -> str:
    candidates = [
        requested_model,
        requested_model.rstrip("/"),
        requested_model.rstrip("/") + "/",
    ]
    for candidate in candidates:
        if candidate in available_ids:
            return candidate
    if len(available_ids) == 1:
        return next(iter(available_ids))
    return requested_model


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    return curl_json(url, payload, timeout, "POST")


def get_json(url: str, timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    return curl_json(url, None, timeout, "GET")


def is_url_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


def build_execution_environment_check(
    base_url: str,
    media_base_url: str | None,
    project_root: Path,
    models_status: int,
    models_raw: str,
) -> dict[str, Any]:
    probe_url = build_media_probe_url(media_base_url, project_root) if media_base_url else None
    expected_media_ok = bool(probe_url and is_url_reachable(probe_url, timeout=2.0))
    return {
        "python_models_probe": {
            "url": f"{base_url.rstrip('/')}/models",
            "http_status": models_status,
            "ok": models_status == 200,
            "error": "" if models_status == 200 else (models_raw.strip() or "http_request_error"),
        },
        "media_http_probe": {
            "base_url": media_base_url or "",
            "base_url_reachable": bool(media_base_url and is_url_reachable(media_base_url, timeout=2.0)),
            "expected_media_url": probe_url or "",
            "expected_media_reachable": expected_media_ok,
        },
    }


def build_media_probe_url(media_base_url: str, project_root: Path) -> str | None:
    candidates = [
        project_root / "pics" / "720x1280" / "jpg" / "circle.jpg",
        project_root / "video" / "720x1280" / "mp4" / "shapes.mp4",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        relative = candidate.resolve().relative_to(project_root.resolve())
        quoted = "/".join(urllib.parse.quote(part) for part in relative.parts)
        return f"{media_base_url.rstrip('/')}/{quoted}"
    return None


@contextlib.contextmanager
def maybe_start_media_server(
    media_base_url: str | None,
    project_root: Path,
    auto_start: bool,
    warnings: list[str] | None = None,
    observations: dict[str, Any] | None = None,
) -> Any:
    if not media_base_url or not auto_start:
        if observations is not None:
            observations.update(
                {
                    "parent_auto_start_requested": bool(auto_start),
                    "strategy": "not_requested",
                    "parent_bind_attempted": False,
                }
            )
        yield
        return

    parsed = urllib.parse.urlparse(media_base_url)
    host = parsed.hostname or ""
    port = parsed.port or 80

    if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"}:
        if observations is not None:
            observations.update(
                {
                    "parent_auto_start_requested": bool(auto_start),
                    "strategy": "non_local_http_url",
                    "parent_bind_attempted": False,
                }
            )
        yield
        return

    probe_url = build_media_probe_url(media_base_url, project_root)
    if is_url_reachable(media_base_url, timeout=2.0):
        if probe_url is None or is_url_reachable(probe_url, timeout=2.0):
            if observations is not None:
                observations.update(
                    {
                        "parent_auto_start_requested": True,
                        "strategy": "reuse_existing_expected_server",
                        "parent_bind_attempted": False,
                    }
                )
            yield
            return
        if warnings is not None:
            warnings.append(
                f"{media_base_url} is reachable, but it is not serving the expected evaluator media root; "
                "continuing with the existing server and child fallbacks."
            )
        if observations is not None:
            observations.update(
                {
                    "parent_auto_start_requested": True,
                    "strategy": "existing_server_wrong_root",
                    "parent_bind_attempted": False,
                }
            )
        yield
        return

    handler = functools.partial(QuietHTTPRequestHandler, directory=str(project_root))
    try:
        httpd = ReusableThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            if is_url_reachable(media_base_url, timeout=2.0) and (
                probe_url is None or is_url_reachable(probe_url, timeout=2.0)
            ):
                if warnings is not None:
                    warnings.append(
                        f"Media server port {port} is already in use; using the existing reachable server."
                    )
                if observations is not None:
                    observations.update(
                        {
                            "parent_auto_start_requested": True,
                            "strategy": "reuse_existing_expected_server_after_eaddrinuse",
                            "parent_bind_attempted": True,
                        }
                    )
                yield
                return
            if warnings is not None:
                warnings.append(
                    f"Media server port {port} is already in use, but {media_base_url} is not serving the expected evaluator media root."
                )
            if observations is not None:
                observations.update(
                    {
                        "parent_auto_start_requested": True,
                        "strategy": "eaddrinuse_wrong_root",
                        "parent_bind_attempted": True,
                    }
                )
            yield
            return
        if exc.errno in {errno.EPERM, errno.EACCES} or isinstance(exc, PermissionError):
            if warnings is not None:
                warnings.append(
                    f"Media server bind on {host}:{port} was denied; continuing without parent auto-start."
                )
            if observations is not None:
                observations.update(
                    {
                        "parent_auto_start_requested": True,
                        "strategy": "bind_denied",
                        "parent_bind_attempted": True,
                    }
                )
            yield
            return
        raise
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    if observations is not None:
        observations.update(
            {
                "parent_auto_start_requested": True,
                "strategy": "started_parent_server",
                "parent_bind_attempted": True,
            }
        )
    try:
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()


def resolve_case_timeout(case: TestCase, base_timeout: float, video_timeout: float | None) -> float:
    if case.media_type == "video" and case.test_type == "semantic":
        if video_timeout is not None:
            return video_timeout
        return max(base_timeout, 300.0)
    return base_timeout


def extract_model_output(response_json: dict[str, Any] | None) -> str:
    if not response_json:
        return ""
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def matched_groups(output: str, expected_groups: list[list[str]], match_mode: str) -> list[bool]:
    lowered = output.lower()
    if match_mode == "ordered_groups":
        cursor = 0
        matches: list[bool] = []
        for group in expected_groups:
            positions = [lowered.find(keyword.lower(), cursor) for keyword in group]
            valid_positions = [position for position in positions if position >= 0]
            if valid_positions:
                cursor = min(valid_positions) + 1
                matches.append(True)
            else:
                matches.append(False)
        return matches
    return [any(keyword.lower() in lowered for keyword in group) for group in expected_groups]


def classify_semantic_result(http_status: int, output: str, case: TestCase) -> tuple[str, list[bool]]:
    if http_status == 0 or http_status >= 400:
        return "BLOCKED", []
    group_matches = matched_groups(output, case.expected_groups, case.match_mode)
    if all(group_matches):
        return "PASS", group_matches
    return "FAIL", group_matches


def build_result(
    case: TestCase,
    project_root: Path,
    payload: dict[str, Any],
    http_status: int,
    latency: float,
    output: str,
    error: str,
    status: str,
    group_matches: list[bool],
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "suite_name": case.suite_name,
        "media_scale": case.media_scale,
        "media_type": case.media_type,
        "transport_mode": case.transport_mode,
        "transport_impl": case.transport_impl,
        "resolution": case.resolution,
        "format": case.media_format,
        "files": [str(path.resolve().relative_to(project_root.resolve())) for path in case.files],
        "prompt": case.prompt,
        "request_payload": payload,
        "max_completion_tokens": payload["max_completion_tokens"],
        "expected_groups": case.expected_groups,
        "group_matches": group_matches,
        "http_status": http_status,
        "status": status,
        "latency_seconds": round(latency, 3),
        "model_output": output,
        "error": error,
        "test_type": case.test_type,
    }


def run_semantic_case(
    case: TestCase,
    base_url: str,
    model: str,
    timeout: float,
    project_root: Path,
    default_max_tokens: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.content}],
        "temperature": 0,
        "max_completion_tokens": case.max_completion_tokens or default_max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    http_status, response_json, raw_body = post_json(url, payload, timeout)
    latency = time.perf_counter() - start
    output = extract_model_output(response_json)
    status, group_matches = classify_semantic_result(http_status, output, case)
    error = "" if http_status and http_status < 400 else raw_body
    return build_result(case, project_root, payload, http_status, latency, output, error, status, group_matches)


def run_ingestion_case(
    case: TestCase,
    base_url: str,
    model: str,
    timeout: float,
    project_root: Path,
    default_max_tokens: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.content}],
        "temperature": 0,
        "max_completion_tokens": case.max_completion_tokens or default_max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    http_status, response_json, raw_body = post_json(url, payload, timeout)
    latency = time.perf_counter() - start
    output = extract_model_output(response_json)

    if http_status == 200 and response_json and "error" not in response_json and output.strip():
        status = "PASS"
        error = ""
    elif http_status == 0 or http_status >= 400:
        status = "BLOCKED"
        error = raw_body
    else:
        status = "FAIL"
        if response_json and "error" in response_json:
            error = response_json["error"].get("message", str(response_json["error"]))
        elif not output.strip():
            error = "empty model output (possible ingestion failure)"
        else:
            error = raw_body
    return build_result(case, project_root, payload, http_status, latency, output, error, status, [])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def escape_cell(value: Any) -> str:
    text = str(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def fenced_text(value: str) -> str:
    return "````text\n" + value + "\n````"


def summarize_by(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        group = str(result.get(key) or "unknown")
        summary.setdefault(group, {})
        status = str(result.get("status") or "unknown")
        summary[group][status] = summary[group].get(status, 0) + 1
    return summary


def counts_by_suite(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return summarize_by(results, "suite_name")


def summarize_included_test_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in results:
        key = (
            str(result.get("suite_name") or ""),
            str(result.get("category") or ""),
            str(result.get("test_type") or ""),
        )
        item = grouped.setdefault(
            key,
            {
                "suite_name": key[0],
                "category": key[1],
                "test_type": key[2],
                "transport_modes": set(),
                "media_scales": set(),
                "total_cases": 0,
                "PASS": 0,
                "FAIL": 0,
                "BLOCKED": 0,
                "SKIP": 0,
            },
        )
        item["transport_modes"].add(str(result.get("transport_mode") or "unknown"))
        item["media_scales"].add(str(result.get("media_scale") or "unknown"))
        item["total_cases"] += 1
        status = str(result.get("status") or "unknown")
        item[status] = item.get(status, 0) + 1

    normalized = []
    for item in grouped.values():
        normalized.append(
            {
                **item,
                "transport_modes": sorted(item["transport_modes"]),
                "media_scales": sorted(item["media_scales"]),
            }
        )
    return sorted(normalized, key=lambda item: (item["suite_name"], item["category"], item["test_type"]))


def default_root_cause_note(failure_class: str) -> str:
    if failure_class == "pipeline_or_serving_issue":
        return "Request did not complete cleanly; treat as transport, timeout, static media, or serving issue first."
    if failure_class == "model_capability_gap":
        return "Request completed, but the answer or tool-use behavior did not meet the expected capability."
    if failure_class == "output_format_or_extraction_issue":
        return "Model produced output, but it did not reliably collapse to the required short format or protocol."
    if failure_class == "accepted_no_call":
        return "No tool call was produced, but this case is explicitly accepted for small-model fallback behavior."
    if failure_class == "none":
        return "Capability check passed."
    return "Need manual inspection."


def classify_failure(result: dict[str, Any]) -> tuple[str, bool, bool, str]:
    existing_failure_class = str(result.get("failure_class") or "")
    if existing_failure_class and existing_failure_class not in {"", "unclassified"}:
        return (
            existing_failure_class,
            bool(result.get("should_count_as_model_error")),
            bool(result.get("should_count_as_engineering_error")),
            str(result.get("root_cause_note") or default_root_cause_note(existing_failure_class)),
        )

    status = result.get("status")
    error_text = str(result.get("error") or "").lower()
    test_type = result.get("test_type")
    if status == "PASS":
        return ("none", False, False, "Capability check passed.")
    if status in {"BLOCKED", "SKIP"}:
        return (
            "pipeline_or_serving_issue",
            False,
            True,
            "Service preflight blocked execution or the run was intentionally skipped.",
        )
    if status == "FAIL":
        if test_type == "ingestion":
            return (
                "pipeline_or_serving_issue",
                False,
                True,
                "Media ingestion failed; treat as transport, serving, or local media path issue first.",
            )
        if "timeout" in error_text or "http" in error_text or "connection" in error_text:
            return (
                "pipeline_or_serving_issue",
                False,
                True,
                "Semantic request failed due to timeout or transport-level issue before a clean answer was obtained.",
            )
        return (
            "model_capability_gap",
            True,
            False,
            "Media was ingested but semantic expectations were not met; treat as a model capability gap candidate.",
        )
    if status == "ERROR":
        return (
            "pipeline_or_serving_issue",
            False,
            True,
            "Unexpected execution error while building or sending the request.",
        )
    return ("unclassified", False, False, "Need manual inspection.")


def normalize_function_calling_results(payload: dict[str, Any], returncode: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        request_error = str(item.get("request_error") or "")
        failure_class = str(item.get("failure_class") or "none")
        status = "PASS" if bool(item.get("passed")) else ("BLOCKED" if failure_class == "pipeline_or_serving_issue" else "FAIL")
        normalized.append(
            {
                "case_id": f"FC-{item.get('id')}",
                "category": "Function Calling",
                "suite_name": FC_SUITE_NAME,
                "media_scale": "standard",
                "media_type": "text",
                "transport_mode": "n/a",
                "transport_impl": "openai_tools",
                "resolution": "",
                "format": "",
                "files": [],
                "prompt": str(item.get("scene") or item.get("id") or "function-calling"),
                "request_payload": {
                    "expected": item.get("expected"),
                    "scene": item.get("scene"),
                    "tool_call_count": item.get("tool_call_count"),
                },
                "max_completion_tokens": 512,
                "expected_groups": [],
                "group_matches": [],
                "http_status": 0 if request_error else 200,
                "status": status,
                "latency_seconds": 0.0,
                "model_output": str(item.get("message") or ""),
                "error": request_error,
                "test_type": "function_calling",
                "failure_class": failure_class,
                "should_count_as_model_error": bool(item.get("should_count_as_model_error")),
                "should_count_as_engineering_error": bool(item.get("should_count_as_engineering_error")),
                "root_cause_note": default_root_cause_note(failure_class),
                "returncode": returncode,
            }
        )
    return normalized


def build_function_calling_placeholder(status: str, error: str) -> list[dict[str, Any]]:
    failure_class = "pipeline_or_serving_issue" if status in {"BLOCKED", "FAIL"} else "none"
    return [
        {
            "case_id": "FC-SUITE",
            "category": "Function Calling",
            "suite_name": FC_SUITE_NAME,
            "media_scale": "standard",
            "media_type": "text",
            "transport_mode": "n/a",
            "transport_impl": "openai_tools",
            "resolution": "",
            "format": "",
            "files": [],
            "prompt": "function-calling-suite",
            "request_payload": {},
            "max_completion_tokens": 512,
            "expected_groups": [],
            "group_matches": [],
            "http_status": 0,
            "status": status,
            "latency_seconds": 0.0,
            "model_output": "",
            "error": error,
            "test_type": "function_calling",
            "failure_class": failure_class,
            "should_count_as_model_error": False,
            "should_count_as_engineering_error": failure_class == "pipeline_or_serving_issue",
            "root_cause_note": default_root_cause_note(failure_class),
        }
    ]


def extract_json_suffix(stdout: str) -> dict[str, Any] | None:
    lines = stdout.strip().splitlines()
    for idx, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        candidate = "\n".join(lines[idx:])
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def run_function_calling_suite(
    script_path: Path,
    test_file: Path,
    base_url: str,
    model: str,
) -> list[dict[str, Any]]:
    launch = build_python_launch(
        script_path,
        "--endpoint",
        f"{base_url.rstrip('/')}/chat/completions",
        "--model",
        model,
        "--test-file",
        str(test_file),
        "--json",
    )
    cmd = launch["cmd"]
    env = launch.get("env")
    launch_error = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(SCRIPT_DIR.parent.parent), env=env)
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except OSError as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        launch_error = stderr
        returncode = 127 if getattr(exc, "errno", None) == errno.ENOENT or isinstance(exc, FileNotFoundError) else 126
    parsed_payload = extract_json_suffix(stdout)
    if not isinstance(parsed_payload, dict):
        return build_function_calling_placeholder("BLOCKED", launch_error or stderr.strip() or "fc_test produced no JSON output")
    payload = parsed_payload
    normalized = normalize_function_calling_results(payload, returncode)
    if normalized:
        return normalized
    if returncode == 0:
        return build_function_calling_placeholder("SKIP", "fc_test returned no per-case results")
    return build_function_calling_placeholder("BLOCKED", launch_error or stderr.strip() or "fc_test failed without per-case results")


def render_markdown(results: list[dict[str, Any]], preflight: dict[str, Any], summary: dict[str, Any]) -> str:
    service_config = preflight.get("service_config", {})
    ingestion_results = [r for r in results if r.get("test_type") == "ingestion"]
    semantic_results = [r for r in results if r.get("test_type") == "semantic"]
    function_calling_results = [r for r in results if r.get("test_type") == "function_calling"]
    counts_by_suite_map = summary.get("counts_by_suite", {})
    included_test_items = summary.get("included_test_items", [])

    config_items = [
        ("dtype", str(service_config.get("dtype", "未知"))),
        ("enforce-eager", "开启" if service_config.get("enforce_eager", False) else "关闭"),
        ("chunked-prefill", "开启" if service_config.get("chunked_prefill", False) else "关闭"),
        ("async-scheduling", "开启" if service_config.get("async_scheduling", False) else "关闭"),
        ("prefix-caching", "开启" if service_config.get("prefix_caching", False) else "关闭"),
        ("function calling serve", "开启" if service_config.get("function_calling", False) else "关闭"),
        ("transport modes", ", ".join(preflight.get("transport_modes", [])) or "未知"),
        ("full transport matrix", "开启" if preflight.get("full_transport_matrix") else "关闭"),
        ("enabled suites", ", ".join(preflight.get("requested_suite_names", [])) or "未知"),
        ("include function calling suite", "开启" if preflight.get("include_function_calling") else "关闭"),
    ]
    if service_config.get("gpu_memory_utilization") is not None:
        config_items.append(("gpu-memory-utilization", str(service_config["gpu_memory_utilization"])))
    if service_config.get("media_base_url"):
        config_items.append(("media-base-url", str(service_config["media_base_url"])))

    lines = [
        "# Qwen3.5-4B 多模态能力测试 Checklist",
        "",
        "## 0. 服务配置",
        "",
        "| 配置项 | 值 |",
        "|---|---:|",
    ]
    for key, value in config_items:
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "### 服务状态",
            "",
            f"- [{'x' if preflight.get('models_ok') else ' '}] `/v1/models` 可访问",
            f"- [{'x' if preflight.get('model_available') else ' '}] 模型名称可用于请求",
            f"- [{'x' if preflight.get('local_media_present') else ' '}] 本地测试媒体目录存在",
            "",
            "### Suite 汇总",
            "",
            "| Suite | PASS | FAIL | BLOCKED | SKIP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for suite_name, counts in counts_by_suite_map.items():
        lines.append(
            f"| {escape_cell(suite_name)} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} | {counts.get('SKIP', 0)} |"
        )

    runtime_warnings = preflight.get("runtime_warnings", [])
    if runtime_warnings:
        lines.extend(
            [
                "",
                "### Runtime Warnings",
                "",
            ]
        )
        for warning in runtime_warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "### 测试内容",
            "",
            "默认矩阵口径：仅 `图片单图语义理解`、`多图输入理解`、`图文穿插输入理解` 默认覆盖 `local_path/base64/http`；其他 evaluator 项默认只测 `local_path`。如需所有项目都覆盖三种输入方式，请显式开启 full transport matrix。",
            "",
            "| Suite | 测试项 | 类型 | Scale | 传输模式 | Total | PASS | FAIL | BLOCKED | SKIP |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in included_test_items:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_cell(item["suite_name"]),
                    escape_cell(item["category"]),
                    escape_cell(item["test_type"]),
                    escape_cell(", ".join(item["media_scales"])),
                    escape_cell(", ".join(item["transport_modes"])),
                    escape_cell(item["total_cases"]),
                    escape_cell(item.get("PASS", 0)),
                    escape_cell(item.get("FAIL", 0)),
                    escape_cell(item.get("BLOCKED", 0)),
                    escape_cell(item.get("SKIP", 0)),
                ]
            )
            + " |"
        )

    if ingestion_results:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 1. Phase 1 格式读取测试",
                "",
                "检查服务能否正常读取并处理各格式的媒体文件。只要 HTTP 200 + 无错误即 PASS，不校验语义理解。",
                "",
            ]
        )
        for category, cat_counts in summarize_by(ingestion_results, "category").items():
            lines.extend(
                [
                    f"### {category}",
                    "",
                    "| Case | Suite | Scale | 文件 | 传输模式 | 底层实现 | HTTP | 耗时(s) | 结果 | 说明 |",
                    "|---|---|---|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for result in [r for r in ingestion_results if r["category"] == category]:
                files = "<br>".join(result["files"])
                remark = result["error"][:120] if result["error"] else result["model_output"][:80]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            escape_cell(result["case_id"]),
                            escape_cell(result["suite_name"]),
                            escape_cell(result["media_scale"]),
                            escape_cell(files),
                            escape_cell(result["transport_mode"]),
                            escape_cell(result["transport_impl"]),
                            escape_cell(result["http_status"]),
                            escape_cell(result["latency_seconds"]),
                            escape_cell(result["status"]),
                            escape_cell(remark),
                        ]
                    )
                    + " |"
                )
            lines.append("")
            lines.append(f"汇总：PASS={cat_counts.get('PASS', 0)} FAIL={cat_counts.get('FAIL', 0)} BLOCKED={cat_counts.get('BLOCKED', 0)}")
            lines.append("")

    if semantic_results:
        lines.extend(["---", "", "## 2. Phase 2 语义理解测试", ""])
        for category, cat_counts in summarize_by(semantic_results, "category").items():
            lines.extend(
                [
                    f"### {category}",
                    "",
                    "| Case | Suite | Scale | 输入文件 | 传输模式 | 底层实现 | 输出 Token 上限 | 预期命中组 | HTTP | 结果 | 耗时(s) | 备注 |",
                    "|---|---|---|---|---:|---:|---:|---|---:|---|---:|---|",
                ]
            )
            for result in [r for r in semantic_results if r["category"] == category]:
                files = "<br>".join(result["files"])
                expected = "<br>".join("/".join(group[:3]) for group in result["expected_groups"])
                remark = result["model_output"][:160] if result["status"] != "BLOCKED" else result["error"][:160]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            escape_cell(result["case_id"]),
                            escape_cell(result["suite_name"]),
                            escape_cell(result["media_scale"]),
                            escape_cell(files),
                            escape_cell(result["transport_mode"]),
                            escape_cell(result["transport_impl"]),
                            escape_cell(result.get("max_completion_tokens")),
                            escape_cell(expected),
                            escape_cell(result["http_status"]),
                            escape_cell(result["status"]),
                            escape_cell(result["latency_seconds"]),
                            escape_cell(remark),
                        ]
                    )
                    + " |"
                )
            lines.append("")
            lines.append(
                f"汇总：PASS={cat_counts.get('PASS', 0)} FAIL={cat_counts.get('FAIL', 0)} BLOCKED={cat_counts.get('BLOCKED', 0)} SKIP={cat_counts.get('SKIP', 0)}"
            )
            lines.append("")

    if function_calling_results:
        lines.extend(
            [
                "---",
                "",
                "## 3. Function Calling 测试",
                "",
                "| Case | 状态 | 根因归类 | 计为链路问题 | 计为模型问题 | 说明 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for result in function_calling_results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_cell(result["case_id"]),
                        escape_cell(result["status"]),
                        escape_cell(result.get("failure_class", "")),
                        escape_cell("yes" if result.get("should_count_as_engineering_error") else "no"),
                        escape_cell("yes" if result.get("should_count_as_model_error") else "no"),
                        escape_cell((result.get("model_output") or result.get("error") or result.get("root_cause_note") or "")[:200]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 结果汇总",
            "",
            "| 维度 | 项目 | PASS | FAIL | BLOCKED | SKIP |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for suite_name, counts in counts_by_suite_map.items():
        lines.append(
            f"| suite | {escape_cell(suite_name)} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} | {counts.get('SKIP', 0)} |"
        )
    for scale_name, counts in summarize_by(results, "media_scale").items():
        lines.append(
            f"| media_scale | {escape_cell(scale_name)} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} | {counts.get('SKIP', 0)} |"
        )
    for transport_mode, counts in summarize_by(results, "transport_mode").items():
        lines.append(
            f"| transport | {escape_cell(transport_mode)} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} | {counts.get('SKIP', 0)} |"
        )

    failures = [result for result in results if result["status"] in {"FAIL", "BLOCKED"}]
    lines.extend(
        [
            "",
            "## 5. 失败 Case 明细",
            "",
            "| Case | Suite | Scale | 类型 | 状态 | HTTP | 传输模式 | 根因归类 | 判责说明 | 错误/输出 |",
            "|---|---|---|---|---|---:|---|---|---|---|",
        ]
    )
    for result in failures:
        detail = result["error"] or result["model_output"]
        lines.append(
            f"| {escape_cell(result['case_id'])} | {escape_cell(result['suite_name'])} | {escape_cell(result['media_scale'])} | {escape_cell(result['test_type'])} | {escape_cell(result['status'])} | {escape_cell(result['http_status'])} | {escape_cell(result['transport_mode'])} | {escape_cell(result.get('failure_class', ''))} | {escape_cell(result.get('root_cause_note', '')[:160])} | {escape_cell(detail[:300])} |"
        )
    if not failures:
        lines.append("| 无 | - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## 6. 完整 Case 输入与输出",
            "",
            "以下内容用于复现和排查。`请求 Payload` 是发送到 `/v1/chat/completions` 的完整 JSON 输入，包含 `messages`、媒体 URL 或 Base64 数据，以及 `max_completion_tokens`。",
            "",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"### {result['case_id']}",
                "",
                f"- 分类：{result['category']}",
                f"- Suite：{result['suite_name']}",
                f"- Scale：{result['media_scale']}",
                f"- 状态：{result['status']}",
                f"- HTTP：{result['http_status']}",
                f"- 传输模式：{result['transport_mode']} ({TRANSPORT_LABELS.get(result['transport_mode'], result['transport_mode'])})",
                f"- 底层实现：{result['transport_impl']}",
                f"- 耗时(s)：{result['latency_seconds']}",
                f"- 输出 Token 上限：{result.get('max_completion_tokens')}",
                f"- 根因归类：{result.get('failure_class', 'none')}",
                f"- 判责说明：{result.get('root_cause_note', '')}",
                "",
                "#### 请求 Payload",
                "",
                fenced_json(result.get("request_payload", {})),
                "",
                "#### 模型完整输出",
                "",
                fenced_text(result["model_output"]) if result["model_output"] else "（无输出）",
                "",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def filter_cases_by_suite(
    cases: list[TestCase],
    suite_names: list[str] | None,
) -> list[TestCase]:
    if not suite_names:
        return cases
    requested = {name.strip() for name in suite_names if name.strip()}
    return [case for case in cases if case.suite_name in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.5-4B multimodal capability tests.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--json", action="store_true", help="Print the final report JSON to stdout.")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--video-timeout",
        type=float,
        default=None,
        help="Per-request timeout for video semantic cases. Defaults to max(--timeout, 300).",
    )
    parser.add_argument(
        "--auto-start-media-server",
        dest="auto_start_media_server",
        action="store_true",
        help="Auto-start a local static HTTP server for --media-base-url when it points to localhost/127.0.0.1.",
    )
    parser.add_argument(
        "--no-auto-start-media-server",
        dest="auto_start_media_server",
        action="store_false",
        help="Disable the default local static HTTP server auto-start behavior.",
    )
    parser.add_argument(
        "--transport-modes",
        nargs="+",
        default=list(STANDARD_TRANSPORT_MODES),
        help="Transport modes to test. Supported: local_path base64 http",
    )
    parser.add_argument(
        "--full-transport-matrix",
        action="store_true",
        help="Expand every evaluator item to all requested transport modes. By default, only selected Phase 2 image understanding items use the full matrix.",
    )
    parser.add_argument(
        "--include-large-image-smoke",
        dest="include_large_image_smoke",
        action="store_true",
        help="Enable 4K-class large image ingestion + semantic smoke suites.",
    )
    parser.add_argument(
        "--no-large-image-smoke",
        dest="include_large_image_smoke",
        action="store_false",
        help="Disable the large-image smoke suites that are enabled by default.",
    )
    parser.add_argument(
        "--large-image-resolutions",
        nargs="+",
        default=list(LARGE_IMAGE_RESOLUTIONS),
        help="Large image resolutions used by --include-large-image-smoke.",
    )
    parser.add_argument(
        "--suite-names",
        nargs="+",
        default=None,
        help="Optional suite filter. Example: phase1_ingestion_large_image_smoke phase2_semantic_large_image_smoke",
    )
    parser.add_argument(
        "--include-function-calling",
        dest="include_function_calling",
        action="store_true",
        help="Run the bundled function-calling suite as part of the evaluator.",
    )
    parser.add_argument(
        "--no-function-calling",
        dest="include_function_calling",
        action="store_false",
        help="Disable the bundled function-calling suite.",
    )
    parser.add_argument(
        "--function-calling-test-file",
        type=Path,
        default=DEFAULT_FC_TEST_FILE,
        help="Path to the bundled function-calling case definition JSON.",
    )
    parser.add_argument(
        "--dtype",
        default=SERVICE_CONFIG_DEFAULTS["dtype"],
        help="Model precision (bfloat16/float16/float32)",
    )
    parser.add_argument(
        "--chunked-prefill",
        default=SERVICE_CONFIG_DEFAULTS["chunked_prefill"],
        type=parse_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--async-scheduling",
        default=SERVICE_CONFIG_DEFAULTS["async_scheduling"],
        type=parse_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--prefix-caching",
        default=SERVICE_CONFIG_DEFAULTS["prefix_caching"],
        type=parse_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--function-calling",
        default=SERVICE_CONFIG_DEFAULTS["function_calling"],
        type=parse_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=SERVICE_CONFIG_DEFAULTS["gpu_memory_utilization"])
    parser.add_argument(
        "--enforce-eager",
        default=SERVICE_CONFIG_DEFAULTS["enforce_eager"],
        type=parse_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--media-base-url",
        default=DEFAULT_MEDIA_BASE_URL,
        help="Base URL for HTTP mode media access, e.g. http://127.0.0.1:9000",
    )
    parser.set_defaults(
        auto_start_media_server=True,
        include_large_image_smoke=True,
        include_function_calling=True,
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else (project_root / "results")

    ensure_local_media(
        project_root,
        include_large_image_smoke=args.include_large_image_smoke,
        large_image_resolutions=list(args.large_image_resolutions),
    )
    ingestion_cases, semantic_cases, enabled_optional_suites = build_capability_cases(
        project_root=project_root,
        media_base_url=args.media_base_url,
        transport_modes=args.transport_modes,
        include_large_image_smoke=args.include_large_image_smoke,
        large_image_resolutions=args.large_image_resolutions,
        full_transport_matrix=args.full_transport_matrix,
    )
    function_calling_suite_names = [FC_SUITE_NAME] if args.include_function_calling else []
    all_suite_names = sorted({case.suite_name for case in ingestion_cases + semantic_cases} | set(function_calling_suite_names))
    if args.suite_names:
        requested_suites = sorted({name.strip() for name in args.suite_names if name.strip()})
        unknown_suites = [name for name in requested_suites if name not in all_suite_names]
        if unknown_suites:
            raise ValueError(
                f"Unknown suite name(s): {unknown_suites}. Available suites: {all_suite_names}"
            )
        ingestion_cases = filter_cases_by_suite(ingestion_cases, requested_suites)
        semantic_cases = filter_cases_by_suite(semantic_cases, requested_suites)
    else:
        requested_suites = all_suite_names
    run_function_calling = args.include_function_calling and (
        not args.suite_names or FC_SUITE_NAME in requested_suites
    )
    all_cases = ingestion_cases + semantic_cases
    missing_issues = assert_case_files(all_cases)
    if missing_issues:
        raise FileNotFoundError(json.dumps(missing_issues, ensure_ascii=False, indent=2))

    report_json = results_dir / "qwen35_multimodal_capability_report.json"
    report_md = results_dir / "qwen35_multimodal_capability_report.md"

    models_status, models_json, models_raw = get_json(f"{args.base_url.rstrip('/')}/models", timeout=10)
    model_available = False
    resolved_model = args.model
    if models_json and isinstance(models_json.get("data"), list):
        available_ids = {str(item.get("id", "")) for item in models_json["data"]}
        resolved_model = resolve_model_id(args.model, available_ids)
        model_available = resolved_model in available_ids

    preflight = {
        "base_url": args.base_url,
        "model": args.model,
        "resolved_model": resolved_model,
        "models_http_status": models_status,
        "models_ok": models_status == 200,
        "model_available": model_available,
        "models_response": models_json if models_json is not None else models_raw,
        "local_media_present": (project_root / "pics").exists() and (project_root / "video").exists(),
        "transport_modes": list(args.transport_modes),
        "full_transport_matrix": bool(args.full_transport_matrix),
        "enabled_optional_suites": enabled_optional_suites,
        "requested_suite_names": requested_suites,
        "include_function_calling": run_function_calling,
        "service_config": {
            "dtype": args.dtype,
            "chunked_prefill": args.chunked_prefill,
            "async_scheduling": args.async_scheduling,
            "prefix_caching": args.prefix_caching,
            "function_calling": args.function_calling,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "media_base_url": args.media_base_url or "N/A (local_path + base64 only)",
        },
    }
    preflight["execution_environment_check"] = build_execution_environment_check(
        args.base_url,
        args.media_base_url,
        project_root,
        models_status,
        models_raw,
    )

    run_started_at = time.perf_counter()
    should_auto_start_media_server = args.auto_start_media_server and not args.dry_run
    runtime_warnings: list[str] = []
    media_server_observation: dict[str, Any] = {}
    with maybe_start_media_server(
        args.media_base_url,
        project_root,
        should_auto_start_media_server,
        runtime_warnings,
        media_server_observation,
    ):
        if args.dry_run:
            results = []
            for case in all_cases:
                payload = {
                    "model": resolved_model,
                    "messages": [{"role": "user", "content": case.content}],
                    "temperature": 0,
                    "max_completion_tokens": case.max_completion_tokens or args.max_tokens,
                    "stream": False,
                }
                results.append(
                    build_result(
                        case,
                        project_root,
                        payload,
                        http_status=0,
                        latency=0.0,
                        output="",
                        error="dry-run",
                        status="SKIP",
                        group_matches=[],
                    )
                )
            if run_function_calling:
                results.extend(build_function_calling_placeholder("SKIP", "dry-run"))
        elif models_status != 200:
            results = []
            for case in all_cases:
                payload = {
                    "model": resolved_model,
                    "messages": [{"role": "user", "content": case.content}],
                    "temperature": 0,
                    "max_completion_tokens": case.max_completion_tokens or args.max_tokens,
                    "stream": False,
                }
                results.append(
                    build_result(
                        case,
                        project_root,
                        payload,
                        http_status=models_status,
                        latency=0.0,
                        output="",
                        error=f"/v1/models unavailable: {models_raw}",
                        status="BLOCKED",
                        group_matches=[],
                    )
                )
            if run_function_calling:
                results.extend(build_function_calling_placeholder("BLOCKED", f"/v1/models unavailable: {models_raw}"))
        else:
            ingestion_results = [
                run_ingestion_case(
                    case=c,
                    base_url=args.base_url,
                    model=resolved_model,
                    timeout=resolve_case_timeout(c, args.timeout, args.video_timeout),
                    project_root=project_root,
                    default_max_tokens=args.max_tokens,
                )
                for c in ingestion_cases
            ]
            semantic_results = [
                run_semantic_case(
                    case=c,
                    base_url=args.base_url,
                    model=resolved_model,
                    timeout=resolve_case_timeout(c, args.timeout, args.video_timeout),
                    project_root=project_root,
                    default_max_tokens=args.max_tokens,
                )
                for c in semantic_cases
            ]
            results = ingestion_results + semantic_results
            if run_function_calling:
                results.extend(
                    run_function_calling_suite(
                        script_path=Path(__file__).resolve().with_name("fc_test.py"),
                        test_file=args.function_calling_test_file.resolve(),
                        base_url=args.base_url,
                        model=resolved_model,
                    )
                )

    summary_counts: dict[str, int] = {}
    test_type_counts: dict[str, int] = {}
    engineering_errors: list[str] = []
    model_limitations: list[str] = []
    non_pass_cases: list[dict[str, Any]] = []
    for result in results:
        summary_counts[result["status"]] = summary_counts.get(result["status"], 0) + 1
        test_type = str(result.get("test_type") or "unknown")
        test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        failure_class, model_err, eng_err, note = classify_failure(result)
        result["failure_class"] = failure_class
        result["should_count_as_model_error"] = model_err
        result["should_count_as_engineering_error"] = eng_err
        result["root_cause_note"] = note
        if model_err:
            model_limitations.append(str(result.get("case_id")))
        if eng_err:
            engineering_errors.append(str(result.get("case_id")))
        if result.get("status") != "PASS":
            non_pass_cases.append(
                {
                    "case_id": result.get("case_id"),
                    "status": result.get("status"),
                    "test_type": result.get("test_type"),
                    "suite_name": result.get("suite_name"),
                    "media_scale": result.get("media_scale"),
                    "transport_mode": result.get("transport_mode"),
                    "failure_class": failure_class,
                    "root_cause_note": note,
                }
            )

    total_runtime_seconds = round(time.perf_counter() - run_started_at, 3)
    summary = {
        "counts_by_status": summary_counts,
        "counts_by_test_type": test_type_counts,
        "counts_by_suite": counts_by_suite(results),
        "counts_by_media_scale": summarize_by(results, "media_scale"),
        "counts_by_transport_mode": summarize_by(results, "transport_mode"),
        "included_test_items": summarize_included_test_items(results),
        "total_runtime_seconds": total_runtime_seconds,
        "engineering_errors": sorted(engineering_errors),
        "model_limitations": sorted(model_limitations),
        "non_pass_cases": non_pass_cases,
        "known_issue_reclassifications": [
            {
                "name": "video_http_pipeline_issues",
                "rule": "If video/http failures are later proven to come from evaluator media serving or timeout handling and re-run passes, exclude the historical failures from model error counts.",
            }
        ],
    }

    report = {
        "preflight": preflight,
        "results": results,
        "summary": summary,
    }
    preflight["runtime_warnings"] = runtime_warnings
    preflight["execution_environment_check"]["parent_media_server"] = media_server_observation
    write_json(report_json, report)
    report_md.write_text(render_markdown(results, preflight, summary), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        counts = {key: summary_counts.get(key, 0) for key in sorted(summary_counts)}
        print(f"Wrote {report_json}")
        print(f"Wrote {report_md}")
        print(f"Total runtime seconds: {total_runtime_seconds}")
        print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
