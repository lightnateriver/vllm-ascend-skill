from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import json
import mimetypes
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"

# Service feature flags — set via CLI, shown in report header
ServiceConfig = {
    "chunked_prefill": True,
    "async_scheduling": True,
    "prefix_caching": True,
    "function_calling": True,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.7,
    "enforce_eager": True,
}

SHAPE_ORDER = [
    "square",
    "rectangle",
    "rhombus",
    "circle",
    "triangle",
    "cylinder",
    "cube",
]

SHAPE_SYNONYMS = {
    "square": ["square", "正方", "正方形"],
    "rectangle": ["rectangle", "长方", "矩形", "长方形"],
    "rhombus": ["rhombus", "diamond", "菱形", "棱形"],
    "circle": ["circle", "圆形", "圆"],
    "triangle": ["triangle", "三角", "三角形"],
    "cylinder": ["cylinder", "圆柱", "圆柱体"],
    "cube": ["cube", "正方体", "立方体"],
}

COLOR_SYNONYMS = {
    "blue": ["blue", "蓝色", "蓝"],
    "green": ["green", "绿色", "绿"],
}

IMAGE_MIME_OVERRIDES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

VIDEO_MIME_OVERRIDES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


@dataclass
class TestCase:
    case_id: str
    category: str
    media_type: str
    input_mode: str
    prompt: str
    content: list[dict[str, Any]]
    expected_groups: list[list[str]] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    resolution: str = ""
    media_format: str = ""
    max_completion_tokens: int = 512


def mime_type_for(path: Path, media_type: str) -> str:
    suffix = path.suffix.lower()
    if media_type == "image":
        return IMAGE_MIME_OVERRIDES.get(suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    if media_type == "video":
        return VIDEO_MIME_OVERRIDES.get(suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def file_url(path: Path) -> str:
    return path.resolve().as_uri()


def http_url(path: Path, base_url: str, project_root: Path) -> str:
    """Build an HTTP URL for a local file relative to project_root."""
    relative = path.resolve().relative_to(project_root.resolve())
    return f"{base_url.rstrip('/')}/{relative.as_posix()}"


def data_url(path: Path, media_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type_for(path, media_type)};base64,{encoded}"


def image_content_url(path: Path, mode: str, base_url: str = "", project_root: Path | None = None) -> dict[str, Any]:
    if mode == "file_url":
        url = file_url(path)
    elif mode == "http":
        url = http_url(path, base_url, project_root or Path.cwd())
    else:
        url = data_url(path, "image")
    return {"type": "image_url", "image_url": {"url": url}}


def video_content_url(path: Path, mode: str = "file_url", base_url: str = "", project_root: Path | None = None) -> dict[str, Any]:
    if mode == "file_url":
        url = file_url(path)
    elif mode == "http":
        url = http_url(path, base_url, project_root or Path.cwd())
    else:
        url = data_url(path, "video")
    return {"type": "video_url", "video_url": {"url": url}}


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def shape_expected_groups(shape: str) -> list[list[str]]:
    return [SHAPE_SYNONYMS[shape], COLOR_SYNONYMS["blue"], COLOR_SYNONYMS["green"]]


def ordered_shape_expected_groups() -> list[list[str]]:
    return [SHAPE_SYNONYMS[shape] for shape in SHAPE_ORDER]


def build_cases(project_root: Path, media_base_url: str | None = None) -> list[TestCase]:
    cases: list[TestCase] = []
    pics = project_root / "pics"
    videos = project_root / "video"

    prompt_describe = "请描述这张图片中的图形、图形颜色和背景颜色。请简短回答。"
    for extension in ["jpg", "png", "webp", "bmp", "tiff"]:
        path = pics / "720x1280" / extension / f"rectangle.{extension}"
        cases.append(
            TestCase(
                case_id=f"IMG-FILE-{extension.upper()}",
                category="图片单图格式支持：本地文件路径",
                media_type="image",
                input_mode="file_url",
                prompt=prompt_describe,
                content=[text_content(prompt_describe), image_content_url(path, "file_url")],
                expected_groups=shape_expected_groups("rectangle"),
                files=[path],
                resolution="720x1280",
                media_format=extension,
            )
        )

    for extension in ["jpg", "png", "webp", "bmp", "tiff"]:
        path = pics / "720x1280" / extension / f"triangle.{extension}"
        cases.append(
            TestCase(
                case_id=f"IMG-B64-{extension.upper()}",
                category="图片单图格式支持：Base64",
                media_type="image",
                input_mode="base64",
                prompt=prompt_describe,
                content=[text_content(prompt_describe), image_content_url(path, "base64")],
                expected_groups=shape_expected_groups("triangle"),
                files=[path],
                resolution="720x1280",
                media_format=extension,
            )
        )

    for resolution in ["256x512", "720x1280", "1920x1080"]:
        path = pics / resolution / "jpg" / "circle.jpg"
        cases.append(
            TestCase(
                case_id=f"IMG-RES-{resolution}",
                category="图片分辨率支持",
                media_type="image",
                input_mode="file_url",
                prompt=prompt_describe,
                content=[text_content(prompt_describe), image_content_url(path, "file_url")],
                expected_groups=shape_expected_groups("circle"),
                files=[path],
                resolution=resolution,
                media_format="jpg",
            )
        )

    multi_prompt = "请按图片输入顺序列出每张图中的形状名称。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
    multi_paths = [pics / "720x1280" / "jpg" / f"{shape}.jpg" for shape in SHAPE_ORDER]
    cases.append(
        TestCase(
            case_id="IMG-MULTI-7-FILE",
            category="多图输入理解",
            media_type="image",
            input_mode="file_url",
            prompt=multi_prompt,
            content=[text_content(multi_prompt), *[image_content_url(path, "file_url") for path in multi_paths]],
            expected_groups=ordered_shape_expected_groups(),
            files=multi_paths,
            resolution="720x1280",
            media_format="jpg",
            max_completion_tokens=512,
        )
    )
    cases.append(
        TestCase(
            case_id="IMG-MULTI-7-B64",
            category="多图输入理解",
            media_type="image",
            input_mode="base64",
            prompt=multi_prompt,
            content=[text_content(multi_prompt), *[image_content_url(path, "base64") for path in multi_paths]],
            expected_groups=ordered_shape_expected_groups(),
            files=multi_paths,
            resolution="720x1280",
            media_format="jpg",
            max_completion_tokens=512,
        )
    )

    first = pics / "720x1280" / "jpg" / "square.jpg"
    second = pics / "720x1280" / "jpg" / "circle.jpg"
    interleave_prompt = "请分别回答第一张图和第二张图是什么形状。只输出英文逗号分隔列表，不要解释。"
    cases.append(
        TestCase(
            case_id="IMG-INTERLEAVE-2",
            category="文本和多图穿插排列",
            media_type="image",
            input_mode="file_url",
            prompt=interleave_prompt,
            content=[
                text_content("第一张图如下。"),
                image_content_url(first, "file_url"),
                text_content("第二张图如下。"),
                image_content_url(second, "file_url"),
                text_content(interleave_prompt),
            ],
            expected_groups=[SHAPE_SYNONYMS["square"], SHAPE_SYNONYMS["circle"]],
            files=[first, second],
            resolution="720x1280",
            media_format="jpg",
        )
    )

    third = pics / "720x1280" / "jpg" / "triangle.jpg"
    cases.append(
        TestCase(
            case_id="IMG-INTERLEAVE-3",
            category="文本和多图穿插排列",
            media_type="image",
            input_mode="file_url",
            prompt="请按出现顺序回答三张图的形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。",
            content=[
                image_content_url(first, "file_url"),
                text_content("这是第一张。"),
                image_content_url(second, "file_url"),
                text_content("这是第二张。"),
                image_content_url(third, "file_url"),
                text_content("这是第三张。请按出现顺序回答三张图的形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"),
            ],
            expected_groups=[SHAPE_SYNONYMS["square"], SHAPE_SYNONYMS["circle"], SHAPE_SYNONYMS["triangle"]],
            files=[first, second, third],
            resolution="720x1280",
            media_format="jpg",
            max_completion_tokens=512,
        )
    )

    video_prompt = "请按出现顺序列出视频中的所有形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
    for extension in ["mp4", "avi", "mov", "mkv"]:
        path = videos / "720x1280" / extension / f"shapes.{extension}"
        cases.append(
            TestCase(
                case_id=f"VID-FILE-{extension.upper()}",
                category="视频格式支持",
                media_type="video",
                input_mode="file_url",
                prompt=video_prompt,
                content=[text_content(video_prompt), video_content_url(path)],
                expected_groups=ordered_shape_expected_groups(),
                files=[path],
                resolution="720x1280",
                media_format=extension,
                max_completion_tokens=512,
            )
        )

    for resolution in ["720x1280", "1080x1920"]:
        path = videos / resolution / "mp4" / "shapes.mp4"
        cases.append(
            TestCase(
                case_id=f"VID-RES-{resolution}",
                category="视频分辨率支持",
                media_type="video",
                input_mode="file_url",
                prompt=video_prompt,
                content=[text_content(video_prompt), video_content_url(path)],
                expected_groups=ordered_shape_expected_groups(),
                files=[path],
                resolution=resolution,
                media_format="mp4",
                max_completion_tokens=512,
            )
        )

    mp4_video = videos / "720x1280" / "mp4" / "shapes.mp4"
    cases.append(
        TestCase(
            case_id="VID-FIRST-SHAPE",
            category="视频理解细节",
            media_type="video",
            input_mode="file_url",
            prompt="视频中第一个出现的形状是什么？只回答形状名称。",
            content=[text_content("视频中第一个出现的形状是什么？只回答形状名称。"), video_content_url(mp4_video)],
            expected_groups=[SHAPE_SYNONYMS["square"]],
            files=[mp4_video],
            resolution="720x1280",
            media_format="mp4",
        )
    )
    cases.append(
        TestCase(
            case_id="VID-LAST-SHAPE",
            category="视频理解细节",
            media_type="video",
            input_mode="file_url",
            prompt="视频中最后一个出现的形状是什么？只回答形状名称。",
            content=[text_content("视频中最后一个出现的形状是什么？只回答形状名称。"), video_content_url(mp4_video)],
            expected_groups=[SHAPE_SYNONYMS["cube"]],
            files=[mp4_video],
            resolution="720x1280",
            media_format="mp4",
        )
    )
    cases.append(
        TestCase(
            case_id="VID-ORDER-ALL",
            category="视频理解细节",
            media_type="video",
            input_mode="file_url",
            prompt=video_prompt,
            content=[text_content(video_prompt), video_content_url(mp4_video)],
            expected_groups=ordered_shape_expected_groups(),
            files=[mp4_video],
            resolution="720x1280",
            media_format="mp4",
            max_completion_tokens=512,
        )
    )

    # ── HTTP mode semantic variants ──
    if media_base_url:
        prompt_describe = "请描述这张图片中的图形、图形颜色和背景颜色。请简短回答。"
        multi_prompt = "请按图片输入顺序列出每张图中的形状名称。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
        video_prompt = "请按出现顺序列出视频中的所有形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"

        # Single image format — HTTP (5 formats)
        for extension in ["jpg", "png", "webp", "bmp", "tiff"]:
            path = pics / "720x1280" / extension / f"rectangle.{extension}"
            cases.append(
                TestCase(
                    case_id=f"IMG-HTTP-{extension.upper()}",
                    category="图片单图格式支持：HTTP",
                    media_type="image",
                    input_mode="http",
                    prompt=prompt_describe,
                    content=[text_content(prompt_describe), image_content_url(path, "http", media_base_url, project_root)],
                    expected_groups=shape_expected_groups("rectangle"),
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                )
            )

        # Resolution — HTTP (3 resolutions)
        for resolution in ["256x512", "720x1280", "1920x1080"]:
            path = pics / resolution / "jpg" / "circle.jpg"
            cases.append(
                TestCase(
                    case_id=f"IMG-RES-HTTP-{resolution}",
                    category="图片分辨率支持：HTTP",
                    media_type="image",
                    input_mode="http",
                    prompt=prompt_describe,
                    content=[text_content(prompt_describe), image_content_url(path, "http", media_base_url, project_root)],
                    expected_groups=shape_expected_groups("circle"),
                    files=[path],
                    resolution=resolution,
                    media_format="jpg",
                )
            )

        # Multi-image — HTTP (7 images)
        multi_paths = [pics / "720x1280" / "jpg" / f"{shape}.jpg" for shape in SHAPE_ORDER]
        cases.append(
            TestCase(
                case_id="IMG-MULTI-7-HTTP",
                category="多图输入理解：HTTP",
                media_type="image",
                input_mode="http",
                prompt=multi_prompt,
                content=[text_content(multi_prompt), *[image_content_url(path, "http", media_base_url, project_root) for path in multi_paths]],
                expected_groups=ordered_shape_expected_groups(),
                files=multi_paths,
                resolution="720x1280",
                media_format="jpg",
                max_completion_tokens=512,
            )
        )

        # Video format — HTTP (4 formats)
        for extension in ["mp4", "avi", "mov", "mkv"]:
            path = videos / "720x1280" / extension / f"shapes.{extension}"
            cases.append(
                TestCase(
                    case_id=f"VID-HTTP-{extension.upper()}",
                    category="视频格式支持：HTTP",
                    media_type="video",
                    input_mode="http",
                    prompt=video_prompt,
                    content=[text_content(video_prompt), video_content_url(path, "http", media_base_url, project_root)],
                    expected_groups=ordered_shape_expected_groups(),
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                    max_completion_tokens=512,
                )
            )

        # Video resolution — HTTP (2 resolutions)
        for resolution in ["720x1280", "1080x1920"]:
            path = videos / resolution / "mp4" / "shapes.mp4"
            cases.append(
                TestCase(
                    case_id=f"VID-RES-HTTP-{resolution}",
                    category="视频分辨率支持：HTTP",
                    media_type="video",
                    input_mode="http",
                    prompt=video_prompt,
                    content=[text_content(video_prompt), video_content_url(path, "http", media_base_url, project_root)],
                    expected_groups=ordered_shape_expected_groups(),
                    files=[path],
                    resolution=resolution,
                    media_format="mp4",
                    max_completion_tokens=512,
                )
            )

        # Video details — HTTP (first, last, order)
        mp4_video = videos / "720x1280" / "mp4" / "shapes.mp4"
        cases.append(
            TestCase(
                case_id="VID-FIRST-HTTP",
                category="视频理解细节：HTTP",
                media_type="video",
                input_mode="http",
                prompt="视频中第一个出现的形状是什么？只回答形状名称。",
                content=[text_content("视频中第一个出现的形状是什么？只回答形状名称。"), video_content_url(mp4_video, "http", media_base_url, project_root)],
                expected_groups=[SHAPE_SYNONYMS["square"]],
                files=[mp4_video],
                resolution="720x1280",
                media_format="mp4",
            )
        )
        cases.append(
            TestCase(
                case_id="VID-LAST-HTTP",
                category="视频理解细节：HTTP",
                media_type="video",
                input_mode="http",
                prompt="视频中最后一个出现的形状是什么？只回答形状名称。",
                content=[text_content("视频中最后一个出现的形状是什么？只回答形状名称。"), video_content_url(mp4_video, "http", media_base_url, project_root)],
                expected_groups=[SHAPE_SYNONYMS["cube"]],
                files=[mp4_video],
                resolution="720x1280",
                media_format="mp4",
            )
        )
        cases.append(
            TestCase(
                case_id="VID-ORDER-HTTP",
                category="视频理解细节：HTTP",
                media_type="video",
                input_mode="http",
                prompt=video_prompt,
                content=[text_content(video_prompt), video_content_url(mp4_video, "http", media_base_url, project_root)],
                expected_groups=ordered_shape_expected_groups(),
                files=[mp4_video],
                resolution="720x1280",
                media_format="mp4",
                max_completion_tokens=512,
            )
        )

    return cases


def build_ingestion_cases(project_root: Path, media_base_url: str | None = None) -> list[TestCase]:
    """Build ingestion-only test cases — check if service can read the file format.
    
    Unlike build_cases(), these cases only verify that the service can ingest
    the media file (HTTP 200 + no error). No semantic understanding check.
    """
    cases: list[TestCase] = []
    pics = project_root / "pics"
    videos = project_root / "video"

    # Minimal prompt — just enough to trigger media ingestion
    ingest_prompt = "describe"
    ingest_max_tokens = 16

    # Image format ingestion: file_url mode (5 formats × 1 resolution)
    for extension in ["jpg", "png", "webp", "bmp", "tiff"]:
        path = pics / "720x1280" / extension / f"rectangle.{extension}"
        if not path.exists():
            path = pics / "720x1280" / extension / f"triangle.{extension}"
        cases.append(
            TestCase(
                case_id=f"INGEST-IMG-FILE-{extension.upper()}",
                category="图片格式读取",
                media_type="image",
                input_mode="file_url",
                prompt=ingest_prompt,
                content=[text_content(ingest_prompt), image_content_url(path, "file_url")],
                files=[path],
                resolution="720x1280",
                media_format=extension,
                max_completion_tokens=ingest_max_tokens,
            )
        )

    # Image format ingestion: base64 mode (5 formats)
    for extension in ["jpg", "png", "webp", "bmp", "tiff"]:
        path = pics / "720x1280" / extension / f"triangle.{extension}"
        if not path.exists():
            path = pics / "720x1280" / extension / f"rectangle.{extension}"
        cases.append(
            TestCase(
                case_id=f"INGEST-IMG-B64-{extension.upper()}",
                category="图片格式读取",
                media_type="image",
                input_mode="base64",
                prompt=ingest_prompt,
                content=[text_content(ingest_prompt), image_content_url(path, "base64")],
                files=[path],
                resolution="720x1280",
                media_format=extension,
                max_completion_tokens=ingest_max_tokens,
            )
        )

    # Image resolution ingestion (3 resolutions × file_url)
    for resolution in ["256x512", "720x1280", "1920x1080"]:
        path = pics / resolution / "jpg" / "circle.jpg"
        cases.append(
            TestCase(
                case_id=f"INGEST-IMG-RES-{resolution}",
                category="图片格式读取",
                media_type="image",
                input_mode="file_url",
                prompt=ingest_prompt,
                content=[text_content(ingest_prompt), image_content_url(path, "file_url")],
                files=[path],
                resolution=resolution,
                media_format="jpg",
                max_completion_tokens=ingest_max_tokens,
            )
        )

    # Video format ingestion (4 formats × file_url)
    for extension in ["mp4", "avi", "mov", "mkv"]:
        path = videos / "720x1280" / extension / f"shapes.{extension}"
        cases.append(
            TestCase(
                case_id=f"INGEST-VID-FILE-{extension.upper()}",
                category="视频格式读取",
                media_type="video",
                input_mode="file_url",
                prompt=ingest_prompt,
                content=[text_content(ingest_prompt), video_content_url(path)],
                files=[path],
                resolution="720x1280",
                media_format=extension,
                max_completion_tokens=ingest_max_tokens,
            )
        )

    # Video resolution ingestion (2 resolutions × mp4)
    for resolution in ["720x1280", "1080x1920"]:
        path = videos / resolution / "mp4" / "shapes.mp4"
        cases.append(
            TestCase(
                case_id=f"INGEST-VID-RES-{resolution}",
                category="视频格式读取",
                media_type="video",
                input_mode="file_url",
                prompt=ingest_prompt,
                content=[text_content(ingest_prompt), video_content_url(path)],
                files=[path],
                resolution=resolution,
                media_format="mp4",
                max_completion_tokens=ingest_max_tokens,
            )
        )

    # ── HTTP mode ingestion variants ──
    if media_base_url:
        # Image format ingestion: http mode (5 formats × 1 resolution)
        for extension in ["jpg", "png", "webp", "bmp", "tiff"]:
            path = pics / "720x1280" / extension / f"rectangle.{extension}"
            if not path.exists():
                path = pics / "720x1280" / extension / f"triangle.{extension}"
            cases.append(
                TestCase(
                    case_id=f"INGEST-IMG-HTTP-{extension.upper()}",
                    category="图片格式读取",
                    media_type="image",
                    input_mode="http",
                    prompt=ingest_prompt,
                    content=[text_content(ingest_prompt), image_content_url(path, "http", media_base_url, project_root)],
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                    max_completion_tokens=ingest_max_tokens,
                )
            )

        # Image resolution ingestion: http mode (3 resolutions)
        for resolution in ["256x512", "720x1280", "1920x1080"]:
            path = pics / resolution / "jpg" / "circle.jpg"
            cases.append(
                TestCase(
                    case_id=f"INGEST-IMG-RES-HTTP-{resolution}",
                    category="图片格式读取",
                    media_type="image",
                    input_mode="http",
                    prompt=ingest_prompt,
                    content=[text_content(ingest_prompt), image_content_url(path, "http", media_base_url, project_root)],
                    files=[path],
                    resolution=resolution,
                    media_format="jpg",
                    max_completion_tokens=ingest_max_tokens,
                )
            )

        # Video format ingestion: http mode (4 formats)
        for extension in ["mp4", "avi", "mov", "mkv"]:
            path = videos / "720x1280" / extension / f"shapes.{extension}"
            cases.append(
                TestCase(
                    case_id=f"INGEST-VID-HTTP-{extension.upper()}",
                    category="视频格式读取",
                    media_type="video",
                    input_mode="http",
                    prompt=ingest_prompt,
                    content=[text_content(ingest_prompt), video_content_url(path, "http", media_base_url, project_root)],
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                    max_completion_tokens=ingest_max_tokens,
                )
            )

        # Video resolution ingestion: http mode (2 resolutions)
        for resolution in ["720x1280", "1080x1920"]:
            path = videos / resolution / "mp4" / "shapes.mp4"
            cases.append(
                TestCase(
                    case_id=f"INGEST-VID-RES-HTTP-{resolution}",
                    category="视频格式读取",
                    media_type="video",
                    input_mode="http",
                    prompt=ingest_prompt,
                    content=[text_content(ingest_prompt), video_content_url(path, "http", media_base_url, project_root)],
                    files=[path],
                    resolution=resolution,
                    media_format="mp4",
                    max_completion_tokens=ingest_max_tokens,
                )
            )

    return cases


def assert_case_files(cases: list[TestCase]) -> list[dict[str, Any]]:
    issues = []
    for case in cases:
        missing = [str(path) for path in case.files if not path.exists()]
        if missing:
            issues.append({"case_id": case.case_id, "missing_files": missing})
    return issues


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed, body
    except Exception as exc:
        return 0, None, repr(exc)


def get_json(url: str, timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed, body
    except Exception as exc:
        return 0, None, repr(exc)


def is_url_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


@contextlib.contextmanager
def maybe_start_media_server(
    media_base_url: str | None,
    project_root: Path,
    auto_start: bool,
) -> Any:
    if not media_base_url or not auto_start:
        yield
        return

    parsed = urllib.parse.urlparse(media_base_url)
    host = parsed.hostname or ""
    port = parsed.port or 80

    # Only auto-manage a local plain HTTP media server.
    if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"}:
        yield
        return

    # If a server is already alive on this endpoint, reuse it.
    if is_url_reachable(media_base_url, timeout=2.0):
        yield
        return

    handler = functools.partial(QuietHTTPRequestHandler, directory=str(project_root))
    httpd = ReusableThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()


def resolve_case_timeout(case: TestCase, base_timeout: float, video_timeout: float | None) -> float:
    if case.media_type == "video" and not case.case_id.startswith("INGEST-"):
        if video_timeout is not None:
            return video_timeout
        # Video semantic cases are much heavier than simple ingestion probes.
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


def matched_groups(output: str, expected_groups: list[list[str]]) -> list[bool]:
    lowered = output.lower()
    return [any(keyword.lower() in lowered for keyword in group) for group in expected_groups]


def classify_result(http_status: int, output: str, groups: list[list[str]]) -> tuple[str, list[bool]]:
    if http_status == 0 or http_status >= 400:
        return "BLOCKED", []
    group_matches = matched_groups(output, groups)
    if all(group_matches):
        return "PASS", group_matches
    return "FAIL", group_matches


def run_case(case: TestCase, base_url: str, model: str, timeout: float, default_max_tokens: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.content}],
        "temperature": 0,
        "max_completion_tokens": case.max_completion_tokens or default_max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    http_status, response_json, raw_body = post_json(url, payload, timeout)
    latency = time.perf_counter() - start
    output = extract_model_output(response_json)
    status, group_matches = classify_result(http_status, output, case.expected_groups)
    error = "" if http_status and http_status < 400 else raw_body

    return {
        "case_id": case.case_id,
        "category": case.category,
        "media_type": case.media_type,
        "input_mode": case.input_mode,
        "resolution": case.resolution,
        "format": case.media_format,
        "files": [str(path.relative_to(Path.cwd())) for path in case.files],
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
        "test_type": "semantic",
    }


def run_ingestion_case(case: TestCase, base_url: str, model: str, timeout: float, default_max_tokens: int) -> dict[str, Any]:
    """Run an ingestion-only check — verify the service can ingest the media file.
    
    PASS criteria: HTTP 200 + no 'error' key in response JSON + non-empty output.
    FAIL: HTTP error or error field in response.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.content}],
        "temperature": 0,
        "max_completion_tokens": case.max_completion_tokens or default_max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    http_status, response_json, raw_body = post_json(url, payload, timeout)
    latency = time.perf_counter() - start
    output = extract_model_output(response_json)

    # Ingestion PASS: HTTP 200 + no error field + non-empty output
    if http_status == 200 and response_json and "error" not in response_json and output.strip():
        status = "PASS"
        error = ""
    elif http_status == 0 or http_status >= 400:
        status = "BLOCKED"
        error = raw_body
    else:
        status = "FAIL"
        error_msg = ""
        if response_json and "error" in response_json:
            error_msg = response_json["error"].get("message", str(response_json["error"]))
        elif not output.strip():
            error_msg = "empty model output (possible ingestion failure)"
        error = error_msg or raw_body

    return {
        "case_id": case.case_id,
        "category": case.category,
        "media_type": case.media_type,
        "input_mode": case.input_mode,
        "resolution": case.resolution,
        "format": case.media_format,
        "files": [str(path.relative_to(Path.cwd())) for path in case.files],
        "prompt": case.prompt,
        "request_payload": payload,
        "max_completion_tokens": payload["max_completion_tokens"],
        "expected_groups": case.expected_groups,
        "group_matches": [],
        "http_status": http_status,
        "status": status,
        "latency_seconds": round(latency, 3),
        "model_output": output,
        "error": error,
        "test_type": "ingestion",
    }


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


def render_markdown(results: list[dict[str, Any]], preflight: dict[str, Any]) -> str:
    service_config = preflight.get("service_config", {})

    # Build service config table
    config_items = [
        ("dtype", str(service_config.get("dtype", "未知"))),
        ("enforce-eager", "开启"),
        ("chunked-prefill", "开启" if service_config.get("chunked_prefill", False) else "关闭"),
        ("async-scheduling", "开启" if service_config.get("async_scheduling", False) else "关闭"),
        ("prefix-caching", "开启" if service_config.get("prefix_caching", False) else "关闭"),
        ("function calling", "开启" if service_config.get("function_calling", False) else "关闭"),
    ]
    if service_config.get("gpu_memory_utilization"):
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
    lines.append("")
    lines.append("### 服务状态")
    lines.append("")
    lines.append(f"- [{'x' if preflight.get('models_ok') else ' '}] `/v1/models` 可访问")
    lines.append(f"- [{'x' if preflight.get('model_available') else ' '}] 模型名称可用于请求")
    lines.append(f"- [{'x' if preflight.get('local_media_present') else ' '}] 本地测试媒体目录存在")
    lines.append("")

    # Split results into ingestion and semantic
    ingestion_results = [r for r in results if r.get("test_type") == "ingestion"]
    semantic_results = [r for r in results if r.get("test_type") != "ingestion"]

    # ── Ingestion results section ──
    if ingestion_results:
        ingest_categories = []
        for result in ingestion_results:
            if result["category"] not in ingest_categories:
                ingest_categories.append(result["category"])

        lines.extend(["---", "", "## 1. 媒体格式读取测试", "",
                       "检查服务能否正常读取并处理各格式的媒体文件。只要 HTTP 200 + 无错误即 PASS，不校验语义理解。",
                       ""])

        for cat in ingest_categories:
            cat_results = [r for r in ingestion_results if r["category"] == cat]
            lines.extend([
                f"### {cat}",
                "",
                "| Case | 文件 | 输入方式 | HTTP | 耗时(s) | 结果 | 说明 |",
                "|---|---|---:|---:|---:|---:|---|",
            ])
            for r in cat_results:
                files = "<br>".join(r["files"])
                remark = r["error"][:120] if r["error"] else r["model_output"][:80]
                lines.append(
                    "| " + " | ".join([
                        escape_cell(r["case_id"]),
                        escape_cell(files),
                        escape_cell(r.get("input_mode", "")),
                        escape_cell(r["http_status"]),
                        escape_cell(r["latency_seconds"]),
                        escape_cell(r["status"]),
                        escape_cell(remark),
                    ]) + " |"
                )
            lines.append("")

        # Ingestion summary matrix
        ingest_summary: dict[str, dict[str, int]] = {}
        for r in ingestion_results:
            ingest_summary.setdefault(r["category"], {"PASS": 0, "FAIL": 0, "BLOCKED": 0})
            ingest_summary[r["category"]][r["status"]] = ingest_summary[r["category"]].get(r["status"], 0) + 1

        lines.extend([
            "### 格式读取汇总",
            "",
            "| 能力项 | PASS | FAIL | BLOCKED |",
            "|---|---:|---:|---:|",
        ])
        for cat, counts in ingest_summary.items():
            lines.append(
                f"| {escape_cell(cat)} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Semantic results section ──

    categories = []
    for result in semantic_results:
        if result["category"] not in categories:
            categories.append(result["category"])

    if categories:
        lines.append("## 2. 语义理解测试")
        lines.append("")

    for category in categories:
        lines.extend(
            [
                f"## {category}",
                "",
                "| Case | 输入文件 | 输出 Token 上限 | 预期命中组 | HTTP | 结果 | 耗时(s) | 备注 |",
                "|---|---|---:|---|---:|---|---:|---|",
            ]
        )
        for result in [item for item in semantic_results if item["category"] == category]:
            files = "<br>".join(result["files"])
            expected = "<br>".join("/".join(group[:3]) for group in result["expected_groups"])
            remark = result["model_output"][:160] if result["status"] != "BLOCKED" else result["error"][:160]
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_cell(result["case_id"]),
                        escape_cell(files),
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

    summary: dict[str, dict[str, int]] = {}
    for result in semantic_results:
        summary.setdefault(result["category"], {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "SKIP": 0})
        summary[result["category"]][result["status"]] = summary[result["category"]].get(result["status"], 0) + 1

    lines.extend(
        [
            "## 3. 语义理解汇总",
            "",
            "| 能力项 | PASS | FAIL | BLOCKED | SKIP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, counts in summary.items():
        lines.append(
            f"| {escape_cell(category)} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | "
            f"{counts.get('BLOCKED', 0)} | {counts.get('SKIP', 0)} |"
        )

    failures = [result for result in semantic_results if result["status"] in {"FAIL", "BLOCKED"}]
    lines.extend(
        [
            "",
            "## 失败 Case 明细",
            "",
            "| Case | 状态 | HTTP | 错误/输出 |",
            "|---|---|---:|---|",
        ]
    )
    for result in failures:
        detail = result["error"] or result["model_output"]
        lines.append(
            f"| {escape_cell(result['case_id'])} | {result['status']} | {result['http_status']} | "
            f"{escape_cell(detail[:300])} |"
        )
    if not failures:
        lines.append("| 无 | - | - | - |")

    lines.extend(
        [
            "",
            "## 完整 Case 输入与输出",
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
                f"- 状态：{result['status']}",
                f"- HTTP：{result['http_status']}",
                f"- 耗时(s)：{result['latency_seconds']}",
                f"- 输出 Token 上限：{result.get('max_completion_tokens')}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.5-4B multimodal capability tests.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
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
        action="store_true",
        help="Auto-start a local static HTTP server for --media-base-url when it points to localhost/127.0.0.1.",
    )
    # Service config flags (for report header)
    parser.add_argument("--dtype", default=ServiceConfig["dtype"], help="Model precision (bfloat16/float16/float32)")
    parser.add_argument("--chunked-prefill", default=ServiceConfig["chunked_prefill"], type=lambda x: x.lower() in ("true", "1", "yes"), nargs="?", const=True)
    parser.add_argument("--async-scheduling", default=ServiceConfig["async_scheduling"], type=lambda x: x.lower() in ("true", "1", "yes"), nargs="?", const=True)
    parser.add_argument("--prefix-caching", default=ServiceConfig["prefix_caching"], type=lambda x: x.lower() in ("true", "1", "yes"), nargs="?", const=True)
    parser.add_argument("--function-calling", default=ServiceConfig["function_calling"], type=lambda x: x.lower() in ("true", "1", "yes"), nargs="?", const=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=ServiceConfig["gpu_memory_utilization"])
    parser.add_argument("--enforce-eager", default=ServiceConfig["enforce_eager"], type=lambda x: x.lower() in ("true", "1", "yes"), nargs="?", const=True)
    parser.add_argument("--media-base-url", default=None, help="Base URL for HTTP mode media access, e.g. http://127.0.0.1:9000")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else (project_root / "results")

    # Build ingestion cases + semantic cases
    ingestion_cases = build_ingestion_cases(project_root, args.media_base_url)
    semantic_cases = build_cases(project_root, args.media_base_url)
    all_cases = ingestion_cases + semantic_cases
    missing_issues = assert_case_files(all_cases)
    if missing_issues:
        raise FileNotFoundError(json.dumps(missing_issues, ensure_ascii=False, indent=2))

    report_json = results_dir / "qwen35_multimodal_capability_report.json"
    report_md = results_dir / "qwen35_multimodal_capability_report.md"

    models_status, models_json, models_raw = get_json(f"{args.base_url.rstrip('/')}/models", timeout=10)
    model_available = False
    if models_json and isinstance(models_json.get("data"), list):
        available_ids = {str(item.get("id", "")) for item in models_json["data"]}
        model_available = args.model in available_ids or bool(available_ids)

    preflight = {
        "base_url": args.base_url,
        "model": args.model,
        "models_http_status": models_status,
        "models_ok": models_status == 200,
        "model_available": model_available,
        "models_response": models_json if models_json is not None else models_raw,
        "local_media_present": (project_root / "pics").exists() and (project_root / "video").exists(),
        "service_config": {
            "dtype": args.dtype,
            "chunked_prefill": args.chunked_prefill,
            "async_scheduling": args.async_scheduling,
            "prefix_caching": args.prefix_caching,
            "function_calling": args.function_calling,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "media_base_url": args.media_base_url or "N/A (file_url + base64 only)",
        },
    }

    def classify_failure(result: dict) -> tuple[str, bool, bool, str]:
        status = result.get("status")
        error_text = str(result.get("error") or "").lower()
        case_id = str(result.get("case_id") or "")
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
        if status in {"ERROR"}:
            return (
                "pipeline_or_serving_issue",
                False,
                True,
                "Unexpected execution error while building or sending the request.",
            )
        return (
            "unclassified",
            False,
            False,
            "Need manual inspection.",
        )

    with maybe_start_media_server(args.media_base_url, project_root, args.auto_start_media_server):
        if args.dry_run:
            results = [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "media_type": case.media_type,
                    "input_mode": case.input_mode,
                    "resolution": case.resolution,
                    "format": case.media_format,
                    "files": [str(path.relative_to(project_root)) for path in case.files],
                    "prompt": case.prompt,
                    "request_payload": {
                        "model": args.model,
                        "messages": [{"role": "user", "content": case.content}],
                        "temperature": 0,
                        "max_completion_tokens": case.max_completion_tokens or args.max_tokens,
                        "stream": False,
                    },
                    "max_completion_tokens": case.max_completion_tokens or args.max_tokens,
                    "expected_groups": case.expected_groups,
                    "group_matches": [],
                    "http_status": None,
                    "status": "SKIP",
                    "latency_seconds": None,
                    "model_output": "",
                    "error": "dry-run",
                    "test_type": "ingestion" if case.case_id.startswith("INGEST-") else "semantic",
                }
                for case in all_cases
            ]
        elif models_status != 200:
            results = [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "media_type": case.media_type,
                    "input_mode": case.input_mode,
                    "resolution": case.resolution,
                    "format": case.media_format,
                    "files": [str(path.relative_to(project_root)) for path in case.files],
                    "prompt": case.prompt,
                    "request_payload": {
                        "model": args.model,
                        "messages": [{"role": "user", "content": case.content}],
                        "temperature": 0,
                        "max_completion_tokens": case.max_completion_tokens or args.max_tokens,
                        "stream": False,
                    },
                    "max_completion_tokens": case.max_completion_tokens or args.max_tokens,
                    "expected_groups": case.expected_groups,
                    "group_matches": [],
                    "http_status": models_status,
                    "status": "BLOCKED",
                    "latency_seconds": None,
                    "model_output": "",
                    "error": f"/v1/models unavailable: {models_raw}",
                    "test_type": "ingestion" if case.case_id.startswith("INGEST-") else "semantic",
                }
                for case in all_cases
            ]
        else:
            ingestion_results = [
                run_ingestion_case(c, args.base_url, args.model, resolve_case_timeout(c, args.timeout, args.video_timeout), args.max_tokens)
                for c in ingestion_cases
            ]
            semantic_results = [
                run_case(c, args.base_url, args.model, resolve_case_timeout(c, args.timeout, args.video_timeout), args.max_tokens)
                for c in semantic_cases
            ]
            results = ingestion_results + semantic_results

    report = {
        "preflight": preflight,
        "results": results,
    }
    summary_counts: dict[str, int] = {}
    test_type_counts: dict[str, int] = {}
    engineering_errors: list[str] = []
    model_limitations: list[str] = []
    non_pass_cases: list[dict] = []
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
                    "failure_class": failure_class,
                    "root_cause_note": note,
                }
            )
    report["summary"] = {
        "counts_by_status": summary_counts,
        "counts_by_test_type": test_type_counts,
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
    write_json(report_json, report)
    report_md.write_text(render_markdown(results, preflight), encoding="utf-8")

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
