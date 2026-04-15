#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WRAPPER_LABELS = {
    "HfRenderer.render_messages_async",
    "parse_chat_messages_async",
    "InputProcessingContext.call_hf_processor",
    "Qwen3VLMultiModalProcessor._call_hf_processor",
    "Qwen3VLProcessor.__call__",
    "Qwen2VLImageProcessorFast.__call__",
    "Qwen2VLImageProcessorFast.preprocess",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def select_profiled_result(driver: dict[str, Any]) -> dict[str, Any] | None:
    results = driver.get("results")
    if not isinstance(results, list):
        results = driver.get("per_round")
    if not isinstance(results, list):
        return None

    for item in results:
        if item.get("profiled"):
            return item

    profile_round = driver.get("profile_round")
    for item in results:
        if item.get("round") == profile_round:
            return item
    return None


def find_outer_span(data: dict[str, Any]) -> dict[str, Any] | None:
    for item in data.get("custom_spans") or []:
        if item.get("label") == "api_server_total.render_messages_async_to_copy_to_buffer":
            return item
    return None


def build_timeline(data: dict[str, Any]) -> str:
    ordered = data.get("ordered_functions") or []
    lines = []
    outer = find_outer_span(data)
    if outer is not None:
        lines.append(
            "API server preprocess chain: "
            f"0.000 -> {format_ms(outer.get('duration_ms'))} ms  [wall-critical-path]"
        )
        lines.append(
            "scope: "
            f"{outer.get('start_function')} -> {outer.get('end_function')}"
        )
    else:
        lines.append("API server preprocess chain: unavailable")

    for item in ordered:
        start_ms = item.get("first_start_ms")
        end_ms = item.get("last_end_ms")
        wall_ms = item.get("wall_span_ms")
        total_ms = item.get("total_ms")
        calls = item.get("calls")
        label = item.get("function")
        metric_bits = [f"wall-critical-path={format_ms(wall_ms)}"]
        if calls and calls > 1:
            metric_bits.append(f"aggregate-total={format_ms(total_ms)}")
        else:
            metric_bits.append(f"inclusive-total={format_ms(total_ms)}")
        metric_bits.append(f"calls={calls}")
        lines.append(
            f"{format_ms(start_ms):>8} -> {format_ms(end_ms):>8}  "
            f"{label}  [{', '.join(metric_bits)}]"
        )
    return "\n".join(lines)


def build_hotspot_table(data: dict[str, Any], limit: int) -> str:
    functions = list(data.get("functions") or [])
    functions.sort(key=lambda item: item.get("total_ms", 0.0), reverse=True)
    rows = ["| function | calls | total_ms | avg_ms |", "|---|---:|---:|---:|"]
    for stats in functions[:limit]:
        label = stats.get("function")
        rows.append(
            "| "
            + f"{label} | {stats.get('calls', 0)} | "
            + f"{format_ms(stats.get('total_ms'))} | {format_ms(stats.get('avg_ms'))} |"
        )
    return "\n".join(rows)


def build_leaf_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = data.get("ordered_functions") or []
    items = []
    for item in ordered:
        label = item.get("function")
        total_ms = float(item.get("total_ms") or 0.0)
        wall_ms = float(item.get("wall_span_ms") or 0.0)
        calls = int(item.get("calls") or 0)
        if label in WRAPPER_LABELS:
            continue
        if "convert_to_tensors" in str(label) and total_ms < 10.0:
            continue
        score = wall_ms if calls > 1 else total_ms
        enriched = dict(item)
        enriched["_leaf_score_ms"] = score
        items.append(enriched)
    items.sort(key=lambda item: item.get("_leaf_score_ms", 0.0), reverse=True)
    return items


def build_conclusion(data: dict[str, Any]) -> str:
    leaves = build_leaf_candidates(data)
    if not leaves:
        return "API server hotspot conclusion unavailable."
    top = leaves[:3]
    parts = [
        f"{item.get('function')} ({format_ms(item.get('wall_span_ms'))} ms)"
        for item in top
    ]
    outer = find_outer_span(data)
    outer_text = ""
    if outer is not None:
        outer_text = (
            " within the API server preprocess chain "
            f"({format_ms(outer.get('duration_ms'))} ms)"
        )
    return (
        "Main API server bottlenecks are "
        + ", ".join(parts)
        + outer_text
        + "."
    )


def build_total_time_section(data: dict[str, Any], driver_result: dict[str, Any] | None) -> str:
    lines = [
        f"- HTTP total wall time: {format_ms(data.get('elapsed_ms'))} ms",
        f"- API server CPU time: {format_ms(data.get('process_cpu_ms'))} ms",
        "- API server preprocess chain: "
        f"{format_ms(data.get('api_server_total_ms'))} ms",
    ]
    if driver_result is not None:
        lines.append(f"- TTFT: {format_ms(driver_result.get('ttft_ms'))} ms")
        lines.append(f"- E2E: {format_ms(driver_result.get('e2e_ms'))} ms")
    lines.append(
        "- response_start_ms and first_body_ms remain separate from API preprocessing totals."
    )
    return "\n".join(lines)


def build_non_additive_section(data: dict[str, Any]) -> str:
    functions = {item.get("function"): item for item in (data.get("functions") or [])}
    lines = [
        "1. Inclusive wrappers overlap with their child calls and must not be summed.",
        "2. Concurrent functions can have large aggregate totals with small wall-critical spans.",
        "3. Long wall windows do not imply equivalent self time when calls are sparse or nested.",
    ]
    load_stats = functions.get("MediaConnector.load_from_url_async")
    if load_stats:
        wall = load_stats.get("wall_span_ms")
        total = load_stats.get("total_ms")
        lines.append(
            "4. Example: MediaConnector.load_from_url_async shows "
            f"aggregate-total={format_ms(total)} ms but wall-critical-path={format_ms(wall)} ms."
        )
    return "\n".join(lines)


def build_suggestions(data: dict[str, Any]) -> str:
    function_items = data.get("functions") or []
    functions = {item.get("function"): item for item in function_items}
    lines = []

    if "Qwen2VLImageProcessorFast._preprocess" in functions:
        lines.append(
            "- Split image preprocessing further before optimizing. Treat this path as the first API-side target."
        )
    if "ProcessorInputs.get_mm_hashes" in functions:
        lines.append(
            "- Investigate multimodal hash generation separately from the HF processor path."
        )
    if "SingleWriterShmObjectStorage.copy_to_buffer" in functions:
        lines.append(
            "- Review SHM copy volume, buffer layout, and CPU binding because repeated copy_to_buffer calls can stretch the tail window."
        )
    if "MediaConnector.load_from_url_async" in functions:
        lines.append(
            "- Do not prioritize load_from_url_async from aggregate-total alone. Check its wall-critical-path first."
        )
    if not lines:
        lines.append("- No specific optimization hints were inferred from the current artifact set.")
    lines.append("- Keep all experiments outside vllm and vllm-ascend source trees unless the user explicitly changes that constraint.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions-json", type=Path, required=True)
    parser.add_argument("--driver-results", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = load_json(args.functions_json)
    driver_result = None
    if args.driver_results is not None:
        driver = load_json(args.driver_results)
        driver_result = select_profiled_result(driver)

    sections = [
        "## Short Conclusion",
        build_conclusion(data),
        "",
        "## API Timeline",
        "```text",
        build_timeline(data),
        "```",
        "",
        "## Total Time Explanation",
        build_total_time_section(data, driver_result),
        "",
        "## Hotspot Functions",
        build_hotspot_table(data, args.top),
        "",
        "## Why The Times Cannot Be Added Directly",
        build_non_additive_section(data),
        "",
        "## Optimization Suggestions",
        build_suggestions(data),
    ]
    output = "\n".join(sections) + "\n"

    if args.out is not None:
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
