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

TOTAL_STAGE_LABEL = "api_server_total.preprocess_wall"


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


def get_stage_spans(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(data.get("stage_spans") or [])
    rows.sort(
        key=lambda row: (
            row.get("order", 9999),
            float("inf") if row.get("start_ms") is None else row["start_ms"],
        )
    )
    return rows


def get_concurrency_windows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(data.get("concurrency_windows") or [])
    rows.sort(
        key=lambda row: (
            float("inf")
            if row.get("first_start_ms") is None
            else row["first_start_ms"]
        )
    )
    return rows


def get_total_stage(data: dict[str, Any]) -> dict[str, Any] | None:
    for row in get_stage_spans(data):
        if row.get("label") == TOTAL_STAGE_LABEL:
            return row
    for row in data.get("custom_spans") or []:
        if row.get("label") == "api_server_total.render_messages_async_to_copy_to_buffer":
            return {
                "label": TOTAL_STAGE_LABEL,
                "start_ms": row.get("start_ms"),
                "end_ms": row.get("end_ms"),
                "wall_ms": row.get("duration_ms"),
                "start_function": row.get("start_function"),
                "end_function": row.get("end_function"),
            }
    return None


def get_non_total_stages(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in get_stage_spans(data) if row.get("label") != TOTAL_STAGE_LABEL]


def get_function_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(data.get("functions") or [])
    rows.sort(key=lambda row: row.get("total_ms", 0.0), reverse=True)
    return rows


def build_stage_table(data: dict[str, Any]) -> str:
    rows = get_stage_spans(data)
    if not rows:
        return "No stage spans were recorded."
    table = [
        "| stage | start_ms | end_ms | wall_ms | start_function | end_function |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        table.append(
            "| "
            + f"{row.get('label')} | {format_ms(row.get('start_ms'))} | "
            + f"{format_ms(row.get('end_ms'))} | {format_ms(row.get('wall_ms'))} | "
            + f"{row.get('start_function')} | {row.get('end_function')} |"
        )
    return "\n".join(table)


def build_concurrency_table(data: dict[str, Any]) -> str:
    rows = get_concurrency_windows(data)
    if not rows:
        return "No concurrency windows were recorded."
    table = [
        "| label | function | calls | aggregate_total_ms | window_wall_ms |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| "
            + f"{row.get('label')} | {row.get('function')} | {row.get('calls')} | "
            + f"{format_ms(row.get('aggregate_total_ms'))} | {format_ms(row.get('window_wall_ms'))} |"
        )
    return "\n".join(table)


def build_leaf_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = get_function_rows(data)
    concurrent_functions = {
        row.get("function") for row in get_concurrency_windows(data)
    }
    candidates: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("function")
        total_ms = float(row.get("total_ms") or 0.0)
        calls = int(row.get("calls") or 0)
        if label in WRAPPER_LABELS:
            continue
        if label == "MediaConnector.load_from_url_async":
            continue
        if "convert_to_tensors" in str(label) and total_ms < 10.0:
            continue

        enriched = dict(row)
        if label in concurrent_functions:
            enriched["_metric_ms"] = float(row.get("wall_span_ms") or 0.0)
            enriched["_metric_kind"] = "window-wall"
        else:
            enriched["_metric_ms"] = total_ms
            enriched["_metric_kind"] = "inclusive-total"
        if calls > 1 and label not in concurrent_functions:
            enriched["_metric_ms"] = float(row.get("wall_span_ms") or 0.0)
            enriched["_metric_kind"] = "multi-call-wall"
        candidates.append(enriched)

    candidates.sort(key=lambda row: row.get("_metric_ms", 0.0), reverse=True)
    return candidates


def build_leaf_hotspot_table(data: dict[str, Any], limit: int) -> str:
    rows = build_leaf_candidates(data)
    if not rows:
        return "No leaf hotspot candidates were selected."
    table = [
        "| function | calls | metric_kind | metric_ms | total_ms | avg_ms |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        table.append(
            "| "
            + f"{row.get('function')} | {row.get('calls')} | {row.get('_metric_kind')} | "
            + f"{format_ms(row.get('_metric_ms'))} | {format_ms(row.get('total_ms'))} | "
            + f"{format_ms(row.get('avg_ms'))} |"
        )
    return "\n".join(table)


def build_timeline(data: dict[str, Any]) -> str:
    total_stage = get_total_stage(data)
    lines: list[str] = []
    if total_stage is not None:
        lines.append(
            "API server preprocess chain: "
            f"{format_ms(total_stage.get('start_ms'))} -> {format_ms(total_stage.get('end_ms'))} ms  "
            f"[wall-critical-path={format_ms(total_stage.get('wall_ms'))}]"
        )
    else:
        lines.append("API server preprocess chain: unavailable")

    windows = get_concurrency_windows(data)
    notable_functions = {
        row.get("function"): row for row in get_function_rows(data)
    }
    leaf_labels = [
        "Qwen2VLImageProcessorFast._preprocess",
        "ProcessorInputs.get_mm_hashes",
        "Qwen2TokenizerFast.__call__",
        "SingleWriterShmObjectStorage.copy_to_buffer",
    ]

    for stage in get_non_total_stages(data):
        lines.append(
            f"{format_ms(stage.get('start_ms')):>8} -> {format_ms(stage.get('end_ms')):>8}  "
            f"{stage.get('label')}  [wall-critical-path={format_ms(stage.get('wall_ms'))}]"
        )
        for window in windows:
            start_ms = window.get("first_start_ms")
            if start_ms is None:
                continue
            if stage.get("start_ms") <= start_ms <= stage.get("end_ms"):
                lines.append(
                    "  "
                    + f"{format_ms(window.get('first_start_ms')):>8} -> {format_ms(window.get('last_end_ms')):>8}  "
                    + f"{window.get('label')}  "
                    + f"[aggregate-total={format_ms(window.get('aggregate_total_ms'))}, "
                    + f"wall-critical-path={format_ms(window.get('window_wall_ms'))}]"
                )
        for label in leaf_labels:
            row = notable_functions.get(label)
            if row is None or row.get("first_start_ms") is None:
                continue
            if stage.get("start_ms") <= row["first_start_ms"] <= stage.get("end_ms"):
                kind = "aggregate-total" if int(row.get("calls") or 0) > 1 else "inclusive-total"
                lines.append(
                    "  "
                    + f"{format_ms(row.get('first_start_ms')):>8} -> {format_ms(row.get('last_end_ms')):>8}  "
                    + f"{label}  "
                    + f"[{kind}={format_ms(row.get('total_ms'))}]"
                )
    return "\n".join(lines)


def build_conclusion(data: dict[str, Any]) -> str:
    stages = get_non_total_stages(data)
    leaves = build_leaf_candidates(data)
    top_stage = max(stages, key=lambda row: row.get("wall_ms", 0.0)) if stages else None
    top_leafs = leaves[:3]
    parts: list[str] = []
    if top_stage is not None:
        parts.append(
            f"largest stage is {top_stage.get('label')} ({format_ms(top_stage.get('wall_ms'))} ms)"
        )
    if top_leafs:
        parts.append(
            "top leaf hotspots are "
            + ", ".join(
                f"{row.get('function')} ({format_ms(row.get('_metric_ms'))} ms)"
                for row in top_leafs
            )
        )
    if not parts:
        return "API server hotspot conclusion unavailable."
    return "Main API server bottlenecks: " + "; ".join(parts) + "."


def build_total_time_section(data: dict[str, Any], driver_result: dict[str, Any] | None) -> str:
    lines = [
        f"- HTTP total wall time: {format_ms(data.get('elapsed_ms'))} ms",
        f"- API server CPU time: {format_ms(data.get('process_cpu_ms'))} ms",
        f"- API server preprocess wall: {format_ms(data.get('api_server_total_ms'))} ms",
        f"- response_start_ms: {format_ms(data.get('response_start_ms'))} ms",
        f"- first_body_ms: {format_ms(data.get('first_body_ms'))} ms",
    ]
    if driver_result is not None:
        lines.append(f"- TTFT: {format_ms(driver_result.get('ttft_ms'))} ms")
        lines.append(f"- E2E: {format_ms(driver_result.get('e2e_ms'))} ms")
    lines.append("- Keep HTTP, preprocess, and token-response timings as separate views.")
    return "\n".join(lines)


def build_non_additive_section(data: dict[str, Any]) -> str:
    stage_rows = get_non_total_stages(data)
    total_stage = get_total_stage(data)
    lines = [
        "- Stage walls are boundary-defined wall spans; they are not the same thing as function total_ms.",
        "- Inclusive wrappers overlap with their child calls and must not be summed.",
        "- Concurrency windows must be interpreted separately from aggregate totals.",
    ]
    if total_stage is not None and stage_rows:
        stage_sum = sum(float(row.get("wall_ms") or 0.0) for row in stage_rows)
        total_wall = float(total_stage.get("wall_ms") or 0.0)
        lines.append(
            "- Sum(stage-wall)="
            + f"{format_ms(stage_sum)} ms while total preprocess wall={format_ms(total_wall)} ms; "
            + f"the gap={format_ms(total_wall - stage_sum)} ms is stage-to-stage glue time."
        )
    for row in get_concurrency_windows(data):
        lines.append(
            f"- {row.get('function')}: aggregate-total={format_ms(row.get('aggregate_total_ms'))} ms, "
            + f"wall-critical-path={format_ms(row.get('window_wall_ms'))} ms."
        )
    return "\n".join(lines)


def build_suggestions(data: dict[str, Any]) -> str:
    functions = {row.get("function"): row for row in get_function_rows(data)}
    lines: list[str] = []
    if "Qwen2VLImageProcessorFast._preprocess" in functions:
        lines.append("- Split image preprocessing further before optimizing; this remains the first API-side target.")
    if "ProcessorInputs.get_mm_hashes" in functions:
        lines.append("- Investigate multimodal hash generation independently from the HF processor stage.")
    if "SingleWriterShmObjectStorage.copy_to_buffer" in functions:
        lines.append("- Review SHM write count, copy granularity, and CPU binding because SHM copy stretches the tail wall window.")
    if "MediaConnector.load_from_url_async" in functions:
        lines.append("- Use media load window wall, not aggregate total, when deciding whether file loading is a real bottleneck.")
    lines.append("- Keep all profiling and optimization experiments outside vllm and vllm-ascend source trees unless the constraint changes.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions-json", type=Path, required=True)
    parser.add_argument("--driver-results", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = load_json(args.functions_json)
    driver_result = None
    if args.driver_results is not None:
        driver_result = select_profiled_result(load_json(args.driver_results))

    sections = [
        "## Short Conclusion",
        build_conclusion(data),
        "",
        "## Ordered API Timeline",
        "```text",
        build_timeline(data),
        "```",
        "",
        "## Stage Wall Breakdown",
        build_stage_table(data),
        "",
        "## Concurrency Windows",
        build_concurrency_table(data),
        "",
        "## Total Time Explanation",
        build_total_time_section(data, driver_result),
        "",
        "## Leaf Hotspot Functions",
        build_leaf_hotspot_table(data, args.top),
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
