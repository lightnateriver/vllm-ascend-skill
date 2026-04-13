#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def load_payload(
    payload_path: Path,
    model: str | None,
    max_completion_tokens: int | None,
    seed: int | None,
) -> dict[str, Any]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if model is not None:
        payload["model"] = model
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if seed is not None:
        payload["seed"] = seed
    return payload


def stream_request(
    host: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], str, float, float, float]:
    start = time.perf_counter()
    first_token_at = None
    last_token_at = None
    final_response = None
    content_parts: list[str] = []

    with requests.post(
        f"{host}/v1/chat/completions",
        json=payload,
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
    return final_response or {}, "".join(content_parts), start, first_token_at, last_token_at


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    warmup_ids = list(range(args.warmup_rounds))
    test_ids = list(range(args.warmup_rounds, args.rounds))

    per_round = []
    for round_id in range(args.rounds):
        payload_path = args.data_dir / f"round_{round_id}" / "payload.json"
        payload = load_payload(
            payload_path,
            args.model,
            args.max_completion_tokens,
            args.seed,
        )
        response, completion_text, start, first_token_at, last_token_at = stream_request(
            args.host,
            payload,
            args.timeout,
        )
        end = time.perf_counter()
        completion_tokens = len(tokenizer.encode(completion_text, add_special_tokens=False))
        result = {
            "round": round_id,
            "payload_path": str(payload_path),
            "is_warmup": round_id in warmup_ids,
            "e2e_ms": (end - start) * 1000.0,
            "ttft_ms": (first_token_at - start) * 1000.0,
            "tpot_ms": (
                max(last_token_at - first_token_at, 0.0) * 1000.0 / max(completion_tokens - 1, 1)
            ),
            "completion_tokens_observed": completion_tokens,
            "usage": response.get("usage") or {},
            "completion_preview": completion_text[:160],
        }
        per_round.append(result)
        print(
            f"round={round_id} warmup={result['is_warmup']} "
            f"ttft_ms={result['ttft_ms']:.2f} tpot_ms={result['tpot_ms']:.2f} "
            f"e2e_ms={result['e2e_ms']:.2f} tokens={completion_tokens}"
        )

    measured = [item for item in per_round if not item["is_warmup"]]
    ttft_values = [item["ttft_ms"] for item in measured]
    tpot_values = [item["tpot_ms"] for item in measured]
    e2e_values = [item["e2e_ms"] for item in measured]

    output = {
        "label": args.label,
        "host": args.host,
        "data_dir": str(args.data_dir),
        "tokenizer": args.tokenizer,
        "rounds": args.rounds,
        "warmup_rounds": warmup_ids,
        "test_rounds": test_ids,
        "model_override": args.model,
        "max_completion_tokens_override": args.max_completion_tokens,
        "seed_override": args.seed,
        "summary": {
            "ttft_ms": summarize(ttft_values),
            "tpot_ms": summarize(tpot_values),
            "e2e_ms": summarize(e2e_values),
        },
        "per_round": per_round,
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"output_file={args.out}")


if __name__ == "__main__":
    main()
