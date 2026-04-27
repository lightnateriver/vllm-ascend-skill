#!/usr/bin/env python3
"""Run and score the local multi-pics dataset against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("multi-pics-datasets/cases")
DEFAULT_OUTPUT_ROOT = Path("multi-pics-runs")
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 1-to-40 multi-pics precision dataset."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--media-mode",
        choices=["file_url", "base64"],
        default="base64",
        help="How to send images to the endpoint.",
    )
    parser.add_argument("--case", default=None, help="Run only one case id like 01 or 40.")
    parser.add_argument(
        "--case-range",
        default="01:40",
        help="Inclusive case range, format start:end. Ignored when --case is set.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="Wait until both /v1/models and a minimal text chat request return HTTP 200.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for the service to become truly ready.",
    )
    parser.add_argument(
        "--ready-poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds for readiness checks.",
    )
    parser.add_argument(
        "--strict-raw",
        action="store_true",
        help="Require the raw output to exactly match the expected short format.",
    )
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    return parser.parse_args()


def discover_case_dirs(dataset_dir: Path, case_id: str | None, case_range: str) -> list[Path]:
    if case_id:
        target = dataset_dir / f"{int(case_id):02d}"
        if not target.is_dir():
            raise FileNotFoundError(f"Case directory not found: {target}")
        return [target]

    start_str, end_str = case_range.split(":", 1)
    start = int(start_str)
    end = int(end_str)
    if start > end:
        raise ValueError(f"Invalid case range: {case_range}")

    case_dirs = []
    for case_num in range(start, end + 1):
        target = dataset_dir / f"{case_num:02d}"
        if not target.is_dir():
            raise FileNotFoundError(f"Case directory not found: {target}")
        case_dirs.append(target)
    return case_dirs


def load_case(case_dir: Path) -> dict[str, Any]:
    metadata = json.loads((case_dir / "answer.json").read_text(encoding="utf-8"))
    images = sorted(metadata["images"], key=lambda item: item["index"])
    metadata["case_dir"] = str(case_dir.resolve())
    metadata["image_paths"] = [str((case_dir / item["filename"]).resolve()) for item in images]
    return metadata


def make_image_part(image_path: str, media_mode: str) -> dict[str, Any]:
    if media_mode == "file_url":
        return {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
    raw = Path(image_path).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def build_messages(case_meta: dict[str, Any], media_mode: str) -> list[dict[str, Any]]:
    system_prompt = (
        "You are a vision evaluation assistant. "
        "Follow the user instruction exactly. "
        "Output only the final short answer. "
        "Do not explain."
    )
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": case_meta["question"]},
    ]
    for image_path in case_meta["image_paths"]:
        user_content.append(make_image_part(image_path, media_mode))

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def exact_raw_match(raw_text: str, question_type: str, expected: str) -> bool:
    cleaned = raw_text.strip()
    if question_type == "yes_no":
        return cleaned.upper() == expected.upper()
    if question_type == "index":
        return cleaned == expected
    return False


def extract_prediction(raw_text: str, question_type: str, image_count: int) -> tuple[str, str]:
    text = raw_text.strip()
    if not text:
        return "UNKNOWN", "empty"

    if question_type == "yes_no":
        match = re.search(r"\b(YES|NO)\b", text, flags=re.IGNORECASE)
        if not match:
            return "UNKNOWN", "format_error"
        return match.group(1).upper(), "ok"

    if question_type == "index":
        match = re.search(r"\b([0-9]+)\b", text)
        if not match:
            return "UNKNOWN", "format_error"
        value = match.group(1)
        if not (1 <= int(value) <= image_count):
            return "UNKNOWN", "out_of_range"
        return value, "ok"

    return "UNKNOWN", "unsupported_question_type"


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        latency_sec = time.perf_counter() - started
        response_body = response.read().decode("utf-8")
    return json.loads(response_body), latency_sec


def endpoint_root(endpoint: str) -> str:
    suffix = "/v1/chat/completions"
    if endpoint.endswith(suffix):
        return endpoint[: -len(suffix)]
    return endpoint.rsplit("/", 1)[0]


def get_json(url: str, timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body.strip() else None
            return response.status, parsed, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = json.loads(body) if body.strip() else None
        return exc.code, parsed, body
    except Exception as exc:  # noqa: BLE001
        return 0, None, repr(exc)


def check_ready_models(base_url: str, timeout: float) -> tuple[bool, str]:
    status, _, body = get_json(f"{base_url}/v1/models", timeout)
    if status == 200:
        return True, "models_ok"
    detail = body.strip() or f"http_{status or 'request_error'}"
    return False, f"models_not_ready:{detail[:160]}"


def check_ready_chat(endpoint: str, model: str, timeout: float) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: hello"}],
        "temperature": 0,
        "max_completion_tokens": 16,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response_json, _ = post_json(endpoint, payload, timeout)
        content = (
            response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if content.strip():
            return True, "chat_ok"
        return False, "chat_empty"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        return False, f"chat_http_{exc.code}:{body[:160]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"chat_request_error:{repr(exc)[:160]}"


def wait_until_ready(
    endpoint: str,
    model: str,
    per_request_timeout: float,
    ready_timeout: float,
    poll_interval: float,
) -> None:
    base_url = endpoint_root(endpoint)
    deadline = time.monotonic() + ready_timeout
    last_reason = "not_checked"
    while time.monotonic() < deadline:
        models_ok, models_reason = check_ready_models(base_url, per_request_timeout)
        chat_ok = False
        chat_reason = "chat_skipped"
        if models_ok:
            chat_ok, chat_reason = check_ready_chat(endpoint, model, per_request_timeout)
        if models_ok and chat_ok:
            print(
                f"[READY  ] models={models_reason} chat={chat_reason}",
                file=sys.stderr,
            )
            return
        last_reason = f"{models_reason}; {chat_reason}"
        print(f"[WAIT   ] {last_reason}", file=sys.stderr)
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Service did not become ready within {ready_timeout:.1f}s. Last status: {last_reason}"
    )


def run_case(
    endpoint: str,
    model: str,
    case_meta: dict[str, Any],
    max_completion_tokens: int,
    temperature: float,
    timeout: float,
    strict_raw: bool,
    media_mode: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": build_messages(case_meta, media_mode),
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result: dict[str, Any] = {
        "case_id": f"{case_meta['case_id']:02d}",
        "image_count": case_meta["image_count"],
        "question_type": case_meta["question_type"],
        "question": case_meta["question"],
        "gold_answer": case_meta["answer"],
        "target": case_meta["target"],
        "images": case_meta["images"],
        "status": "unknown",
        "error_type": "unknown",
        "raw_prediction": "",
        "extracted_prediction": "UNKNOWN",
        "strict_raw_ok": False,
        "latency_sec": None,
    }
    try:
        response_json, latency_sec = post_json(endpoint, payload, timeout)
        result["latency_sec"] = round(latency_sec, 3)
        raw_prediction = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        result["raw_prediction"] = raw_prediction
        result["finish_reason"] = response_json.get("choices", [{}])[0].get("finish_reason")
        extracted, extract_status = extract_prediction(
            raw_prediction,
            case_meta["question_type"],
            case_meta["image_count"],
        )
        result["extracted_prediction"] = extracted
        result["strict_raw_ok"] = exact_raw_match(
            raw_prediction, case_meta["question_type"], case_meta["answer"]
        )
        if extracted == "UNKNOWN":
            result["status"] = "unknown"
            result["error_type"] = extract_status
        elif strict_raw and not result["strict_raw_ok"]:
            result["status"] = "wrong"
            result["error_type"] = "strict_raw_mismatch"
        elif extracted == case_meta["answer"]:
            result["status"] = "correct"
            result["error_type"] = "ok"
        else:
            result["status"] = "wrong"
            result["error_type"] = "wrong_answer"
    except urllib.error.HTTPError as exc:
        result["status"] = "unknown"
        result["error_type"] = f"http_{exc.code}"
        body = exc.read().decode("utf-8", errors="replace")
        result["raw_prediction"] = body
    except urllib.error.URLError as exc:
        reason_text = str(exc.reason).lower()
        if "timed out" in reason_text:
            result["status"] = "timeout"
            result["error_type"] = "timeout"
        else:
            result["status"] = "unknown"
            result["error_type"] = "request_error"
        result["raw_prediction"] = str(exc)
    except TimeoutError:
        result["status"] = "timeout"
        result["error_type"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "unknown"
        result["error_type"] = "request_error"
        result["raw_prediction"] = repr(exc)
    return result


def summarize(results: list[dict[str, Any]], run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    counts = {
        "correct": sum(1 for item in results if item["status"] == "correct"),
        "wrong": sum(1 for item in results if item["status"] == "wrong"),
        "unknown": sum(1 for item in results if item["status"] == "unknown"),
        "timeout": sum(1 for item in results if item["status"] == "timeout"),
    }
    summary = {
        "run_name": run_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "endpoint": args.endpoint,
        "model": args.model,
        "total": len(results),
        **counts,
        "accuracy": round(counts["correct"] / len(results), 4) if results else 0.0,
        "case_ids": [item["case_id"] for item in results],
        "failed_cases": [
            item["case_id"] for item in results if item["status"] in {"wrong", "unknown", "timeout"}
        ],
    }
    return summary


def write_outputs(run_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for item in results:
        (run_dir / f"{item['case_id']}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (run_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "image_count",
                "question_type",
                "gold_answer",
                "raw_prediction",
                "extracted_prediction",
                "status",
                "error_type",
                "strict_raw_ok",
                "latency_sec",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "image_count": item["image_count"],
                    "question_type": item["question_type"],
                    "gold_answer": item["gold_answer"],
                    "raw_prediction": item["raw_prediction"],
                    "extracted_prediction": item["extracted_prediction"],
                    "status": item["status"],
                    "error_type": item["error_type"],
                    "strict_raw_ok": item["strict_raw_ok"],
                    "latency_sec": item["latency_sec"],
                }
            )


def build_run_dir(output_dir: str | None, run_name: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    root = DEFAULT_OUTPUT_ROOT
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = run_name or f"multi_pics_{timestamp}"
    return root / name


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    case_dirs = discover_case_dirs(dataset_dir, args.case, args.case_range)
    run_dir = build_run_dir(args.output_dir, args.run_name)

    if args.wait_ready:
        wait_until_ready(
            endpoint=args.endpoint,
            model=args.model,
            per_request_timeout=min(args.timeout, 30.0),
            ready_timeout=args.ready_timeout,
            poll_interval=args.ready_poll_interval,
        )

    results = []
    for case_dir in case_dirs:
        case_meta = load_case(case_dir)
        result = run_case(
            endpoint=args.endpoint,
            model=args.model,
            case_meta=case_meta,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            strict_raw=args.strict_raw,
            media_mode=args.media_mode,
        )
        results.append(result)
        print(
            f"[{result['status'].upper():7}] case={result['case_id']} "
            f"images={result['image_count']} gold={result['gold_answer']} "
            f"pred={result['extracted_prediction']} error={result['error_type']}",
            file=sys.stderr,
        )

    summary = summarize(results, run_dir, args)
    write_outputs(run_dir, results, summary)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            "multi-pics summary: "
            f"total={summary['total']} "
            f"correct={summary['correct']} "
            f"wrong={summary['wrong']} "
            f"unknown={summary['unknown']} "
            f"timeout={summary['timeout']} "
            f"accuracy={summary['accuracy']}",
        )
        print(f"outputs: {run_dir.resolve()}")
    return 0 if summary["wrong"] == 0 and summary["unknown"] == 0 and summary["timeout"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
