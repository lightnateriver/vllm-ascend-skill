#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any

from pyinstrument import Profiler


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sanitize_path(path: str) -> str:
    sanitized = path.strip().replace("/", "_")
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", sanitized)
    return sanitized.strip("_") or "root"


@dataclass(frozen=True)
class ProfileConfig:
    output_dir: Path
    endpoint_pattern: re.Pattern[str] | None
    header_name: str | None
    header_value: str
    max_requests: int
    dump_text: bool
    dump_html: bool
    show_all: bool
    timeline: bool
    interval: float
    dump_function_stats: bool

    @classmethod
    def from_env(cls) -> "ProfileConfig":
        pattern_text = os.getenv(
            "API_SERVER_PROFILE_ENDPOINT_REGEX",
            r"^/v1/chat/completions$",
        ).strip()
        header_name = os.getenv("API_SERVER_PROFILE_HEADER", "x-profile-api-server")
        header_name = header_name.strip().lower() or None
        max_requests = int(os.getenv("API_SERVER_PROFILE_MAX_REQUESTS", "1"))
        return cls(
            output_dir=Path(
                os.getenv("API_SERVER_PROFILE_DIR", "/tmp/api_server_profiles")
            ),
            endpoint_pattern=re.compile(pattern_text) if pattern_text else None,
            header_name=header_name,
            header_value=os.getenv("API_SERVER_PROFILE_HEADER_VALUE", "1"),
            max_requests=max_requests,
            dump_text=_env_flag("API_SERVER_PROFILE_DUMP_TEXT", True),
            dump_html=_env_flag("API_SERVER_PROFILE_DUMP_HTML", True),
            show_all=_env_flag("API_SERVER_PROFILE_SHOW_ALL", True),
            timeline=_env_flag("API_SERVER_PROFILE_TIMELINE", False),
            interval=float(os.getenv("API_SERVER_PROFILE_INTERVAL", "0.0001")),
            dump_function_stats=_env_flag("API_SERVER_PROFILE_DUMP_FUNCTIONS", True),
        )


@dataclass
class FunctionTiming:
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    first_start_ms: float | None = None
    last_end_ms: float | None = None


_REQUEST_FUNCTION_STATS: contextvars.ContextVar[dict[str, FunctionTiming] | None] = (
    contextvars.ContextVar("request_function_stats", default=None)
)
_REQUEST_FUNCTION_EVENTS: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("request_function_events", default=None)
)
_REQUEST_START_TIME: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "request_start_time",
    default=None,
)
_REQUEST_CUSTOM_SPANS: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("request_custom_spans", default=None)
)

_API_SERVER_TOTAL_SPAN_LABEL = (
    "api_server_total.render_messages_async_to_copy_to_buffer"
)
_API_SERVER_TOTAL_START_LABEL = "HfRenderer.render_messages_async"
_API_SERVER_TOTAL_END_LABEL = "SingleWriterShmObjectStorage.copy_to_buffer"


def _relative_request_ms(ts: float) -> float | None:
    request_start = _REQUEST_START_TIME.get()
    if request_start is None:
        return None
    return (ts - request_start) * 1000.0


def _record_function_timing(label: str, started_at: float, ended_at: float) -> None:
    elapsed_ms = (ended_at - started_at) * 1000.0
    start_ms = _relative_request_ms(started_at)
    end_ms = _relative_request_ms(ended_at)

    stats = _REQUEST_FUNCTION_STATS.get()
    if stats is not None:
        timing = stats.get(label)
        if timing is None:
            timing = FunctionTiming()
            stats[label] = timing

        timing.calls += 1
        timing.total_ms += elapsed_ms
        timing.max_ms = max(timing.max_ms, elapsed_ms)
        if start_ms is not None and (
            timing.first_start_ms is None or start_ms < timing.first_start_ms
        ):
            timing.first_start_ms = start_ms
        if end_ms is not None and (
            timing.last_end_ms is None or end_ms > timing.last_end_ms
        ):
            timing.last_end_ms = end_ms

    events = _REQUEST_FUNCTION_EVENTS.get()
    if events is not None:
        events.append(
            {
                "function": label,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": elapsed_ms,
            }
        )

    custom_spans = _REQUEST_CUSTOM_SPANS.get()
    if custom_spans is not None:
        api_server_total = custom_spans.setdefault(
            _API_SERVER_TOTAL_SPAN_LABEL,
            {
                "label": _API_SERVER_TOTAL_SPAN_LABEL,
                "start_ms": None,
                "end_ms": None,
                "duration_ms": None,
                "start_function": _API_SERVER_TOTAL_START_LABEL,
                "end_function": _API_SERVER_TOTAL_END_LABEL,
            },
        )
        if label == _API_SERVER_TOTAL_START_LABEL and api_server_total["start_ms"] is None:
            api_server_total["start_ms"] = start_ms
        if label == _API_SERVER_TOTAL_END_LABEL:
            api_server_total["end_ms"] = end_ms
        if (
            api_server_total["start_ms"] is not None
            and api_server_total["end_ms"] is not None
        ):
            api_server_total["duration_ms"] = (
                api_server_total["end_ms"] - api_server_total["start_ms"]
            )


def _wrap_callable(obj: Any, attr: str, label: str) -> None:
    original = getattr(obj, attr, None)
    if original is None or getattr(original, "_api_server_profile_wrapped", False):
        return

    if not callable(original):
        return

    if getattr(original, "__wrapped__", None) is not None and getattr(
        original, "_api_server_profile_wrapped", False
    ):
        return

    if hasattr(original, "__call__") and asyncio.iscoroutinefunction(original):
        @functools.wraps(original)
        async def async_wrapper(*args, **kwargs):
            started_at = time.perf_counter()
            try:
                return await original(*args, **kwargs)
            finally:
                _record_function_timing(label, started_at, time.perf_counter())

        async_wrapper._api_server_profile_wrapped = True  # type: ignore[attr-defined]
        setattr(obj, attr, async_wrapper)
        return

    @functools.wraps(original)
    def sync_wrapper(*args, **kwargs):
        started_at = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _record_function_timing(label, started_at, time.perf_counter())

    sync_wrapper._api_server_profile_wrapped = True  # type: ignore[attr-defined]
    setattr(obj, attr, sync_wrapper)


def _install_function_timing_patches() -> None:
    import transformers.feature_extraction_utils as hf_feature_utils
    import transformers.tokenization_utils_base as hf_tokenization_base
    import transformers.models.qwen2.tokenization_qwen2_fast as hf_qwen2_tok_fast
    import transformers.models.qwen2_vl.image_processing_qwen2_vl_fast as hf_qwen2_vl_img_fast
    import transformers.models.qwen3_vl.processing_qwen3_vl as hf_qwen3_processing
    import vllm.model_executor.models.qwen3_vl as qwen3_vl_model
    import vllm.distributed.device_communicators.shm_object_storage as shm_storage
    import vllm.multimodal.media.connector as media_connector
    import vllm.multimodal.processing.context as mm_context
    import vllm.multimodal.processing.inputs as mm_inputs
    import vllm.utils.async_utils as async_utils
    import vllm.renderers.hf as renderer_hf

    targets = [
        (
            renderer_hf.HfRenderer,
            "render_messages_async",
            "HfRenderer.render_messages_async",
        ),
        (renderer_hf, "parse_chat_messages_async", "parse_chat_messages_async"),
        (renderer_hf, "safe_apply_chat_template", "safe_apply_chat_template"),
        (
            async_utils.AsyncMicrobatchTokenizer,
            "encode",
            "AsyncMicrobatchTokenizer.encode",
        ),
        (
            media_connector.MediaConnector,
            "load_from_url_async",
            "MediaConnector.load_from_url_async",
        ),
        (
            mm_context.InputProcessingContext,
            "call_hf_processor",
            "InputProcessingContext.call_hf_processor",
        ),
        (
            qwen3_vl_model.Qwen3VLMultiModalProcessor,
            "_call_hf_processor",
            "Qwen3VLMultiModalProcessor._call_hf_processor",
        ),
        (
            hf_qwen3_processing.Qwen3VLProcessor,
            "__call__",
            "Qwen3VLProcessor.__call__",
        ),
        (
            hf_qwen2_vl_img_fast.Qwen2VLImageProcessorFast,
            "__call__",
            "Qwen2VLImageProcessorFast.__call__",
        ),
        (
            hf_qwen2_vl_img_fast.Qwen2VLImageProcessorFast,
            "preprocess",
            "Qwen2VLImageProcessorFast.preprocess",
        ),
        (
            hf_qwen2_vl_img_fast.Qwen2VLImageProcessorFast,
            "_preprocess",
            "Qwen2VLImageProcessorFast._preprocess",
        ),
        (
            hf_qwen2_tok_fast.Qwen2TokenizerFast,
            "__call__",
            "Qwen2TokenizerFast.__call__",
        ),
        (
            hf_feature_utils.BatchFeature,
            "convert_to_tensors",
            "BatchFeature.convert_to_tensors",
        ),
        (
            hf_tokenization_base.BatchEncoding,
            "convert_to_tensors",
            "BatchEncoding.convert_to_tensors",
        ),
        (
            mm_inputs.ProcessorInputs,
            "get_mm_hashes",
            "ProcessorInputs.get_mm_hashes",
        ),
        (
            shm_storage.SingleWriterShmObjectStorage,
            "copy_to_buffer",
            "SingleWriterShmObjectStorage.copy_to_buffer",
        ),
    ]

    for obj, attr, label in targets:
        _wrap_callable(obj, attr, label)


def install_monkey_patch() -> None:
    import vllm.entrypoints.openai.api_server as api_server

    if getattr(api_server, "_api_server_profile_monkey_patch_installed", False):
        return

    config = ProfileConfig.from_env()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = count(1)
    original_build_app = api_server.build_app
    _install_function_timing_patches()

    class WholeRequestProfilerApp:
        def __init__(self, app: Any) -> None:
            self.app = app
            self.state = app.state
            self.profiled_requests = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self.app, name)

        def _should_profile(self, scope: dict[str, Any]) -> bool:
            if scope.get("type") != "http":
                return False

            path = scope.get("path", "")
            if config.endpoint_pattern and not config.endpoint_pattern.search(path):
                return False

            headers = {
                key.decode("latin1").lower(): value.decode("latin1")
                for key, value in scope.get("headers", [])
            }
            if config.header_name:
                return headers.get(config.header_name) == config.header_value

            if config.max_requests >= 0 and self.profiled_requests >= config.max_requests:
                return False
            return True

        async def __call__(self, scope, receive, send) -> None:
            if not self._should_profile(scope):
                await self.app(scope, receive, send)
                return

            if config.header_name is None:
                self.profiled_requests += 1

            request_no = next(sequence)
            started_at = time.perf_counter()
            wall_ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            method = scope.get("method", "NA")
            path = scope.get("path", "")
            query_string = scope.get("query_string", b"").decode("latin1")
            client = scope.get("client")
            client_repr = f"{client[0]}:{client[1]}" if client else "unknown"
            status_code: int | None = None
            response_start_ms: float | None = None
            first_body_ms: float | None = None
            process_cpu_start = time.process_time()

            async def send_wrapper(message) -> None:
                nonlocal status_code, response_start_ms, first_body_ms
                now_ms = (time.perf_counter() - started_at) * 1000.0
                if message.get("type") == "http.response.start":
                    status_code = int(message.get("status", 0))
                    if response_start_ms is None:
                        response_start_ms = now_ms
                elif message.get("type") == "http.response.body":
                    if first_body_ms is None:
                        body = message.get("body", b"")
                        if body or not message.get("more_body", False):
                            first_body_ms = now_ms
                await send(message)

            profiler = Profiler(async_mode="enabled", interval=config.interval)
            request_function_stats: dict[str, FunctionTiming] | None = (
                {} if config.dump_function_stats else None
            )
            request_function_events: list[dict[str, Any]] | None = (
                [] if config.dump_function_stats else None
            )
            request_custom_spans: dict[str, dict[str, Any]] | None = (
                {} if config.dump_function_stats else None
            )
            stats_token = _REQUEST_FUNCTION_STATS.set(request_function_stats)
            events_token = _REQUEST_FUNCTION_EVENTS.set(request_function_events)
            request_start_token = _REQUEST_START_TIME.set(started_at)
            custom_spans_token = _REQUEST_CUSTOM_SPANS.set(request_custom_spans)
            profiler.start()
            failure: BaseException | None = None
            try:
                await self.app(scope, receive, send_wrapper)
            except BaseException as exc:
                failure = exc
                raise
            finally:
                profiler.stop()
                request_function_events = _REQUEST_FUNCTION_EVENTS.get()
                request_custom_spans = _REQUEST_CUSTOM_SPANS.get()
                _REQUEST_FUNCTION_STATS.reset(stats_token)
                _REQUEST_FUNCTION_EVENTS.reset(events_token)
                _REQUEST_START_TIME.reset(request_start_token)
                _REQUEST_CUSTOM_SPANS.reset(custom_spans_token)
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                process_cpu_ms = (time.process_time() - process_cpu_start) * 1000.0
                status_label = str(status_code if status_code is not None else "NA")
                stem = (
                    f"{wall_ts}_{request_no:04d}_{method}_{_sanitize_path(path)}_"
                    f"{status_label}"
                )
                html_path = config.output_dir / f"{stem}.html"
                text_path = config.output_dir / f"{stem}.txt"
                functions_json_path = config.output_dir / f"{stem}.functions.json"
                functions_txt_path = config.output_dir / f"{stem}.functions.txt"

                if config.dump_html:
                    html_path.write_text(profiler.output_html(), encoding="utf-8")
                if config.dump_text:
                    text_path.write_text(
                        profiler.output_text(
                            unicode=False,
                            color=False,
                            show_all=config.show_all,
                            timeline=config.timeline,
                        ),
                        encoding="utf-8",
                    )
                api_server_total_ms: float | None = None
                if config.dump_function_stats and request_function_stats is not None:
                    function_rows = [
                        {
                            "function": label,
                            **asdict(timing),
                            "wall_span_ms": (
                                timing.last_end_ms - timing.first_start_ms
                                if timing.first_start_ms is not None
                                and timing.last_end_ms is not None
                                else None
                            ),
                            "avg_ms": (
                                timing.total_ms / timing.calls if timing.calls else 0.0
                            ),
                        }
                        for label, timing in request_function_stats.items()
                    ]
                    ordered_function_rows = sorted(
                        function_rows,
                        key=lambda row: (
                            float("inf")
                            if row["first_start_ms"] is None
                            else row["first_start_ms"]
                        ),
                    )
                    function_rows.sort(key=lambda row: row["total_ms"], reverse=True)
                    if request_function_events is None:
                        request_function_events = []
                    request_function_events.sort(
                        key=lambda row: (
                            float("inf")
                            if row["start_ms"] is None
                            else row["start_ms"]
                        ),
                    )
                    custom_span_rows = (
                        sorted(
                            request_custom_spans.values(),
                            key=lambda row: (
                                float("inf")
                                if row["start_ms"] is None
                                else row["start_ms"]
                            ),
                        )
                        if request_custom_spans is not None
                        else []
                    )
                    api_server_total_ms = next(
                        (
                            row["duration_ms"]
                            for row in custom_span_rows
                            if row["label"] == _API_SERVER_TOTAL_SPAN_LABEL
                        ),
                        None,
                    )
                    functions_json_path.write_text(
                        json.dumps(
                            {
                                "request_no": request_no,
                                "method": method,
                                "path": path,
                                "status": status_code,
                                "elapsed_ms": elapsed_ms,
                                "process_cpu_ms": process_cpu_ms,
                                "response_start_ms": response_start_ms,
                                "first_body_ms": first_body_ms,
                                "api_server_total_ms": api_server_total_ms,
                                "custom_spans": custom_span_rows,
                                "ordered_functions": ordered_function_rows,
                                "function_events": request_function_events,
                                "functions": function_rows,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    functions_txt_path.write_text(
                        (
                            "custom_spans\n"
                            + "\n".join(
                                [
                                    (
                                        f"{row['label']}: start_ms={row['start_ms']:.3f} "
                                        f"end_ms={row['end_ms']:.3f} "
                                        f"duration_ms={row['duration_ms']:.3f}"
                                    )
                                    for row in custom_span_rows
                                    if row["start_ms"] is not None
                                    and row["end_ms"] is not None
                                    and row["duration_ms"] is not None
                                ]
                            )
                            + "\n\nfunctions_by_total_ms\n"
                            + "\n".join(
                                [
                                    (
                                        f"{row['function']}: calls={row['calls']} "
                                        f"total_ms={row['total_ms']:.3f} "
                                        f"avg_ms={row['avg_ms']:.3f} "
                                        f"max_ms={row['max_ms']:.3f} "
                                        f"first_start_ms={row['first_start_ms']:.3f} "
                                        f"last_end_ms={row['last_end_ms']:.3f} "
                                        f"wall_span_ms={row['wall_span_ms']:.3f}"
                                    )
                                    for row in function_rows
                                    if row["first_start_ms"] is not None
                                    and row["last_end_ms"] is not None
                                    and row["wall_span_ms"] is not None
                                ]
                            )
                            + "\n\nfunctions_in_order\n"
                            + "\n".join(
                                [
                                    (
                                        f"{row['function']}: first_start_ms={row['first_start_ms']:.3f} "
                                        f"last_end_ms={row['last_end_ms']:.3f} "
                                        f"wall_span_ms={row['wall_span_ms']:.3f} "
                                        f"aggregate_total_ms={row['total_ms']:.3f} "
                                        f"calls={row['calls']}"
                                    )
                                    for row in ordered_function_rows
                                    if row["first_start_ms"] is not None
                                    and row["last_end_ms"] is not None
                                    and row["wall_span_ms"] is not None
                                ]
                            )
                            + "\n"
                        ),
                        encoding="utf-8",
                    )

                print(
                    "[api-server-profiler] "
                    f"request_no={request_no} method={method} path={path} "
                    f"query={query_string or '-'} status={status_label} "
                    f"elapsed_ms={elapsed_ms:.2f} client={client_repr} "
                    f"process_cpu_ms={process_cpu_ms:.2f} "
                    f"api_server_total_ms={api_server_total_ms if config.dump_function_stats and request_function_stats is not None and api_server_total_ms is not None else -1:.2f} "
                    f"response_start_ms={response_start_ms if response_start_ms is not None else -1:.2f} "
                    f"first_body_ms={first_body_ms if first_body_ms is not None else -1:.2f} "
                    f"html={html_path if config.dump_html else '-'} "
                    f"text={text_path if config.dump_text else '-'} "
                    f"functions={functions_json_path if config.dump_function_stats else '-'} "
                    f"exception={type(failure).__name__ if failure else '-'}",
                    flush=True,
                )

    def patched_build_app(*args, **kwargs):
        app = original_build_app(*args, **kwargs)
        return WholeRequestProfilerApp(app)

    api_server.build_app = patched_build_app
    api_server._api_server_profile_monkey_patch_installed = True

    print(
        "[api-server-profiler] monkey patch installed "
        f"output_dir={config.output_dir} "
        f"endpoint_regex={config.endpoint_pattern.pattern if config.endpoint_pattern else '<all>'} "
        f"header={config.header_name or '<none>'} "
        f"header_value={config.header_value} "
        f"max_requests={config.max_requests} "
        f"interval={config.interval}",
        flush=True,
    )


def main() -> None:
    install_monkey_patch()
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
