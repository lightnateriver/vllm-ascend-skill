#!/usr/bin/env python3
"""Run and score the local multi-pics dataset against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from media_input_utils import build_image_reference, curl_json_request, local_http_media_server, resolve_model_id

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
        choices=["base64", "local_path", "http"],
        default="base64",
        help="How to send images to the endpoint.",
    )
    parser.add_argument(
        "--media-base-url",
        default="",
        help="Base URL for local HTTP image serving, for example http://127.0.0.1:9000 .",
    )
    parser.add_argument(
        "--media-root",
        default="",
        help="Local media root for http path mapping. Required for http mode.",
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


def build_messages(
    case_meta: dict[str, Any],
    media_mode: str,
    media_root: str,
    media_base_url: str,
) -> list[dict[str, Any]]:
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
        image_ref, _ = build_image_reference(
            image_b64=None,
            image_path=image_path,
            media_mode=media_mode,
            media_root=media_root,
            media_base_url=media_base_url,
            fallback_name=Path(image_path).name,
        )
        user_content.append({"type": "image_url", "image_url": {"url": image_ref}})

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


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any] | None, float, str]:
    started = time.perf_counter()
    status, parsed, body = curl_json_request(url, payload, timeout, "POST")
    latency_sec = time.perf_counter() - started
    return status, parsed, latency_sec, body


def endpoint_root(endpoint: str) -> str:
    suffix = "/v1/chat/completions"
    if endpoint.endswith(suffix):
        return endpoint[: -len(suffix)]
    return endpoint.rsplit("/", 1)[0]


def get_json(url: str, timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    return curl_json_request(url, None, timeout, "GET")


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
    status, response_json, _, body = post_json(endpoint, payload, timeout)
    if status == 200 and response_json:
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content.strip():
            return True, "chat_ok"
        return False, "chat_empty"
    if status == 0:
        return False, f"chat_request_error:{body[:160]}"
    return False, f"chat_http_{status}:{body[:160]}"


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
    media_root: str,
    media_base_url: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": build_messages(case_meta, media_mode, media_root, media_base_url),
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
        "media_mode": media_mode,
        "status": "unknown",
        "error_type": "unknown",
        "failure_class": "unclassified",
        "should_count_as_model_error": False,
        "should_count_as_engineering_error": False,
        "root_cause_note": "",
        "raw_prediction": "",
        "extracted_prediction": "UNKNOWN",
        "strict_raw_ok": False,
        "latency_sec": None,
    }
    status, response_json, latency_sec, body = post_json(endpoint, payload, timeout)
    try:
        result["latency_sec"] = round(latency_sec, 3)
        raw_prediction = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            if response_json
            else ""
        )
        result["raw_prediction"] = raw_prediction
        if response_json:
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
        if status == 0:
            result["status"] = "timeout"
            result["error_type"] = "timeout" if "timed out" in body.lower() else "request_error"
            result["raw_prediction"] = body
        elif status >= 400:
            result["status"] = "unknown"
            result["error_type"] = f"http_{status}"
            result["raw_prediction"] = body
        elif extracted == "UNKNOWN":
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
    except TimeoutError:
        result["status"] = "timeout"
        result["error_type"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "unknown"
        result["error_type"] = "request_error"
        result["raw_prediction"] = repr(exc)
    if result["status"] == "correct":
        result["failure_class"] = "none"
        result["root_cause_note"] = "Prediction matched the expected short answer."
    elif result["status"] == "wrong":
        result["failure_class"] = "model_capability_gap"
        result["should_count_as_model_error"] = True
        if result["error_type"] == "strict_raw_mismatch":
            result["root_cause_note"] = (
                "Answer extracted correctly only partially or with extra text; output format drift under strict mode."
            )
        else:
            result["root_cause_note"] = (
                "Service returned a decodable answer, but the extracted prediction did not match the gold answer."
            )
    elif result["status"] in {"unknown", "timeout"}:
        if str(result["error_type"]).startswith("http_") or result["error_type"] in {"request_error", "timeout"}:
            result["failure_class"] = "pipeline_or_serving_issue"
            result["should_count_as_engineering_error"] = True
            result["root_cause_note"] = (
                "Request did not complete cleanly due to HTTP, request, or timeout issues; treat as pipeline/serving issue first."
            )
        else:
            result["failure_class"] = "output_format_or_extraction_issue"
            result["root_cause_note"] = (
                "Model output did not reliably collapse to the required short answer format; do not treat as a pure vision failure."
            )
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
        "media_mode": args.media_mode,
        "media_root": args.media_root if args.media_mode != "base64" else "",
        "media_base_url": args.media_base_url,
        "total": len(results),
        **counts,
        "accuracy": round(counts["correct"] / len(results), 4) if results else 0.0,
        "case_ids": [item["case_id"] for item in results],
        "failed_cases": [
            item["case_id"] for item in results if item["status"] in {"wrong", "unknown", "timeout"}
        ],
        "failure_class_counts": {
            "model_capability_gap": sum(1 for item in results if item["failure_class"] == "model_capability_gap"),
            "pipeline_or_serving_issue": sum(1 for item in results if item["failure_class"] == "pipeline_or_serving_issue"),
            "output_format_or_extraction_issue": sum(
                1 for item in results if item["failure_class"] == "output_format_or_extraction_issue"
            ),
            "none": sum(1 for item in results if item["failure_class"] == "none"),
        },
        "engineering_error_cases": [
            item["case_id"] for item in results if item["should_count_as_engineering_error"]
        ],
        "model_limitation_cases": [
            item["case_id"] for item in results if item["should_count_as_model_error"]
        ],
        "wrong_cases_detailed": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "error_type": item["error_type"],
                "failure_class": item["failure_class"],
                "root_cause_note": item["root_cause_note"],
            }
            for item in results
            if item["status"] in {"wrong", "unknown", "timeout"}
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
    base_url = endpoint_root(args.endpoint)
    models_status, models_json, _ = get_json(f"{base_url}/v1/models", min(args.timeout, 30.0))
    resolved_model = args.model
    if models_status == 200 and models_json and isinstance(models_json.get("data"), list):
        available_ids = {str(item.get("id", "")) for item in models_json["data"]}
        resolved_model = resolve_model_id(args.model, available_ids)

    if args.wait_ready:
        wait_until_ready(
            endpoint=args.endpoint,
            model=resolved_model,
            per_request_timeout=min(args.timeout, 30.0),
            ready_timeout=args.ready_timeout,
            poll_interval=args.ready_poll_interval,
        )

    with local_http_media_server(args.media_mode, args.media_root, args.media_base_url) as effective_media_base_url:
        results = []
        for case_dir in case_dirs:
            case_meta = load_case(case_dir)
            result = run_case(
                endpoint=args.endpoint,
                model=resolved_model,
                case_meta=case_meta,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                strict_raw=args.strict_raw,
                media_mode=args.media_mode,
                media_root=args.media_root,
                media_base_url=effective_media_base_url or "",
            )
            results.append(result)
            print(
                f"[{result['status'].upper():7}] case={result['case_id']} "
                f"images={result['image_count']} gold={result['gold_answer']} "
                f"pred={result['extracted_prediction']} error={result['error_type']}",
                file=sys.stderr,
            )

    original_media_base_url = args.media_base_url
    args.media_base_url = effective_media_base_url or args.media_base_url
    summary = summarize(results, run_dir, args)
    args.media_base_url = original_media_base_url
    summary["resolved_model"] = resolved_model
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
