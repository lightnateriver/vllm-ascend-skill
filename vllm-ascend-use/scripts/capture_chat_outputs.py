#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def stream_request(host: str, payload: dict[str, Any], timeout: int) -> tuple[str, dict[str, Any]]:
    content_parts: list[str] = []
    final_response: dict[str, Any] = {}

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
                content_parts.append(content)

    return "".join(content_parts), final_response


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    results = []
    for round_id in range(args.rounds):
        payload_path = args.data_dir / f"round_{round_id}" / "payload.json"
        payload = load_payload(
            payload_path,
            args.model,
            args.max_completion_tokens,
            args.seed,
        )
        text, response = stream_request(args.host, payload, args.timeout)
        results.append(
            {
                "round": round_id,
                "payload": str(payload_path),
                "text": text,
                "response": response,
            }
        )
        preview = text[:80].replace("\n", "\\n")
        print(f"round={round_id} chars={len(text)} preview={preview}")

    output = {
        "label": args.label,
        "host": args.host,
        "rounds": args.rounds,
        "model_override": args.model,
        "max_completion_tokens_override": args.max_completion_tokens,
        "seed_override": args.seed,
        "results": results,
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_file={args.out}")


if __name__ == "__main__":
    main()
