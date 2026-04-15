#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


def wait_until_ready(host: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{host}/v1/models", timeout=10)
            if resp.ok:
                return
            last_error = f"status={resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(3)
    raise RuntimeError(f"service not ready within {timeout}s: {last_error}")


def load_payload(payload_path: Path) -> dict[str, Any]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    return payload


def stream_request(
    host: str,
    payload: dict[str, Any],
    timeout: int,
    headers: dict[str, str] | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_at = None
    last_token_at = None
    content_parts: list[str] = []
    final_response: dict[str, Any] = {}

    with requests.post(
        f"{host}/v1/chat/completions",
        json=payload,
        headers=headers,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue

            data_str = raw_line[6:]
            if data_str == "[DONE]":
                break

            data = json.loads(data_str)
            final_response = data
            choices = data.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                content_parts.append(content)

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    if last_token_at is None:
        last_token_at = end

    usage = final_response.get("usage") or {}
    completion_tokens = max(int(usage.get("completion_tokens") or 0), 1)

    return {
        "response_id": final_response.get("id"),
        "usage": usage,
        "completion_preview": "".join(content_parts)[:200],
        "ttft_ms": (first_token_at - start) * 1000.0,
        "e2e_ms": (end - start) * 1000.0,
        "tpot_ms": max(last_token_at - first_token_at, 0.0) * 1000.0 / max(completion_tokens - 1, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--profile-round", type=int, default=3)
    parser.add_argument("--profile-header", default="x-profile-api-server")
    parser.add_argument("--profile-header-value", default="1")
    parser.add_argument("--wait-ready-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wait_until_ready(args.host, args.wait_ready_timeout)

    results = []
    for round_id in range(args.rounds):
        payload_path = args.data_dir / f"round_{round_id}" / "payload.json"
        payload = load_payload(payload_path)
        headers = None
        if round_id == args.profile_round:
            headers = {args.profile_header: args.profile_header_value}

        result = stream_request(args.host, payload, args.timeout, headers)
        result.update(
            {
                "round": round_id,
                "is_warmup": round_id < args.warmup_rounds,
                "profiled": round_id == args.profile_round,
                "payload_path": str(payload_path),
            }
        )
        results.append(result)
        print(
            f"round={round_id} warmup={result['is_warmup']} profiled={result['profiled']} "
            f"ttft_ms={result['ttft_ms']:.2f} tpot_ms={result['tpot_ms']:.2f} "
            f"e2e_ms={result['e2e_ms']:.2f} prompt_tokens={result['usage'].get('prompt_tokens')} "
            f"completion_tokens={result['usage'].get('completion_tokens')}",
            flush=True,
        )

    output = {
        "host": args.host,
        "data_dir": str(args.data_dir),
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "profile_round": args.profile_round,
        "results": results,
    }
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"output_file={args.out}", flush=True)


if __name__ == "__main__":
    main()
