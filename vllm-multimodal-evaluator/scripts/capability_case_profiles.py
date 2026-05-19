from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


STANDARD_IMAGE_RESOLUTIONS = ("256x512", "720x1280", "1920x1080")
LARGE_IMAGE_RESOLUTIONS = ("4096x4096", "4096x6144", "4096x8192")
STANDARD_VIDEO_RESOLUTIONS = ("720x1280", "1080x1920")
LARGE_VIDEO_RESOLUTIONS = ("4096x4096", "4096x6144", "4096x8192")
IMAGE_FORMATS = ("jpg", "png", "webp", "bmp", "tiff")
VIDEO_FORMATS = ("mp4", "avi", "mov", "mkv")
STANDARD_TRANSPORT_MODES = ("local_path", "base64", "http")
TRANSPORT_IMPL_BY_MODE = {
    "local_path": "file_url",
    "base64": "data_url",
    "http": "http_url",
}
TRANSPORT_LABELS = {
    "local_path": "local_path",
    "base64": "base64",
    "http": "http",
}

SHAPE_ORDER = (
    "square",
    "rectangle",
    "rhombus",
    "circle",
    "triangle",
    "cylinder",
    "cube",
)

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

PROMPT_DESCRIBE_IMAGE = "请描述这张图片中的图形、图形颜色和背景颜色。请简短回答。"
PROMPT_MULTI_IMAGE = "请按图片输入顺序列出每张图中的形状名称。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
PROMPT_INTERLEAVE_TWO = "请分别回答第一张图和第二张图是什么形状。只输出英文逗号分隔列表，不要解释。"
PROMPT_INTERLEAVE_THREE = "请按出现顺序回答三张图的形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
PROMPT_MULTI_VIDEO = "请按输入顺序回答每个视频中第一个出现的形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
PROMPT_VIDEO_ORDER = "请按出现顺序列出视频中的所有形状。只输出英文逗号分隔列表，不要解释，不要编号，不要分析。"
PROMPT_VIDEO_FIRST = "视频中第一个出现的形状是什么？只回答形状名称。"
PROMPT_VIDEO_LAST = "视频中最后一个出现的形状是什么？只回答形状名称。"
INGEST_PROMPT = "describe"
INGEST_MAX_TOKENS = 16
MULTI_VIDEO_SEQUENCE = (
    "square",
    "rectangle",
    "rhombus",
    "circle",
    "triangle",
    "cylinder",
    "cube",
    "square",
    "triangle",
    "cube",
)
PRIMARY_LOCAL_PATH_MODE = ("local_path",)


@dataclass
class TestCase:
    case_id: str
    category: str
    suite_name: str
    media_scale: str
    media_type: str
    transport_mode: str
    transport_impl: str
    prompt: str
    content: list[dict[str, Any]]
    expected_groups: list[list[str]] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    resolution: str = ""
    media_format: str = ""
    max_completion_tokens: int = 512
    match_mode: str = "all_groups"

    @property
    def test_type(self) -> str:
        return "ingestion" if self.suite_name.startswith("phase1_") else "semantic"


def normalize_transport_modes(
    modes: Iterable[str] | None,
    media_base_url: str | None,
) -> list[str]:
    requested = list(modes or STANDARD_TRANSPORT_MODES)
    normalized: list[str] = []
    for mode in requested:
        value = str(mode).strip().lower()
        if value not in TRANSPORT_IMPL_BY_MODE:
            raise ValueError(f"Unsupported transport mode: {mode}")
        if value == "http" and not media_base_url:
            raise ValueError("HTTP transport requires --media-base-url.")
        if value not in normalized:
            normalized.append(value)
    return normalized


def primary_transport_modes(modes: Iterable[str]) -> list[str]:
    normalized = list(modes)
    if "local_path" in normalized:
        return ["local_path"]
    return normalized[:1]


def mime_type_for(path: Path, media_type: str) -> str:
    suffix = path.suffix.lower()
    if media_type == "image":
        return IMAGE_MIME_OVERRIDES.get(suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    if media_type == "video":
        return VIDEO_MIME_OVERRIDES.get(suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def local_path_url(path: Path) -> str:
    return path.resolve().as_uri()


def http_url(path: Path, base_url: str, project_root: Path) -> str:
    relative = path.resolve().relative_to(project_root.resolve())
    return f"{base_url.rstrip('/')}/{relative.as_posix()}"


def data_url(path: Path, media_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type_for(path, media_type)};base64,{encoded}"


def build_media_content(
    path: Path,
    media_type: str,
    transport_mode: str,
    media_base_url: str | None,
    project_root: Path,
) -> dict[str, Any]:
    if transport_mode == "local_path":
        url = local_path_url(path)
    elif transport_mode == "base64":
        url = data_url(path, media_type)
    elif transport_mode == "http":
        if not media_base_url:
            raise ValueError("HTTP transport requested without media_base_url.")
        url = http_url(path, media_base_url, project_root)
    else:
        raise ValueError(f"Unsupported transport mode: {transport_mode}")

    if media_type == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    if media_type == "video":
        return {"type": "video_url", "video_url": {"url": url}}
    raise ValueError(f"Unsupported media type: {media_type}")


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def shape_expected_groups(shape: str) -> list[list[str]]:
    return [SHAPE_SYNONYMS[shape], COLOR_SYNONYMS["blue"], COLOR_SYNONYMS["green"]]


def ordered_shape_expected_groups() -> list[list[str]]:
    return [SHAPE_SYNONYMS[shape] for shape in SHAPE_ORDER]


def _transport_code(transport_mode: str) -> str:
    return {
        "local_path": "FILE",
        "base64": "B64",
        "http": "HTTP",
    }[transport_mode]


def _case(
    *,
    case_id: str,
    category: str,
    suite_name: str,
    media_scale: str,
    media_type: str,
    transport_mode: str,
    prompt: str,
    content: list[dict[str, Any]],
    files: list[Path],
    resolution: str,
    media_format: str,
    max_completion_tokens: int,
    expected_groups: list[list[str]] | None = None,
    match_mode: str = "all_groups",
) -> TestCase:
    return TestCase(
        case_id=case_id,
        category=category,
        suite_name=suite_name,
        media_scale=media_scale,
        media_type=media_type,
        transport_mode=transport_mode,
        transport_impl=TRANSPORT_IMPL_BY_MODE[transport_mode],
        prompt=prompt,
        content=content,
        files=files,
        resolution=resolution,
        media_format=media_format,
        max_completion_tokens=max_completion_tokens,
        expected_groups=expected_groups or [],
        match_mode=match_mode,
    )


def build_standard_ingestion_cases(
    project_root: Path,
    media_base_url: str | None,
    transport_modes: Iterable[str],
    full_transport_matrix: bool = False,
) -> list[TestCase]:
    modes = normalize_transport_modes(transport_modes, media_base_url)
    default_modes = modes if full_transport_matrix else primary_transport_modes(modes)
    pics = project_root / "pics"
    videos = project_root / "video"
    suite_name = "phase1_ingestion_standard"
    cases: list[TestCase] = []

    for transport_mode in default_modes:
        if transport_mode in {"local_path", "base64", "http"}:
            for extension in IMAGE_FORMATS:
                shape_name = "rectangle" if transport_mode != "base64" else "triangle"
                path = pics / "720x1280" / extension / f"{shape_name}.{extension}"
                cases.append(
                    _case(
                        case_id=f"INGEST-IMG-{_transport_code(transport_mode)}-{extension.upper()}",
                        category="图片格式读取",
                        suite_name=suite_name,
                        media_scale="standard",
                        media_type="image",
                        transport_mode=transport_mode,
                        prompt=INGEST_PROMPT,
                        content=[
                            text_content(INGEST_PROMPT),
                            build_media_content(path, "image", transport_mode, media_base_url, project_root),
                        ],
                        files=[path],
                        resolution="720x1280",
                        media_format=extension,
                        max_completion_tokens=INGEST_MAX_TOKENS,
                    )
                )

    for transport_mode in default_modes:
        for resolution in STANDARD_IMAGE_RESOLUTIONS:
            path = pics / resolution / "jpg" / "circle.jpg"
            cases.append(
                _case(
                    case_id=f"INGEST-IMG-RES-{_transport_code(transport_mode)}-{resolution}",
                    category="图片分辨率读取",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="image",
                    transport_mode=transport_mode,
                    prompt=INGEST_PROMPT,
                    content=[
                        text_content(INGEST_PROMPT),
                        build_media_content(path, "image", transport_mode, media_base_url, project_root),
                    ],
                    files=[path],
                    resolution=resolution,
                    media_format="jpg",
                    max_completion_tokens=INGEST_MAX_TOKENS,
                )
            )

    for transport_mode in default_modes:
        for extension in VIDEO_FORMATS:
            path = videos / "720x1280" / extension / f"shapes.{extension}"
            cases.append(
                _case(
                    case_id=f"INGEST-VID-{_transport_code(transport_mode)}-{extension.upper()}",
                    category="视频格式读取",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=INGEST_PROMPT,
                    content=[
                        text_content(INGEST_PROMPT),
                        build_media_content(path, "video", transport_mode, media_base_url, project_root),
                    ],
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                    max_completion_tokens=INGEST_MAX_TOKENS,
                )
            )

    for transport_mode in default_modes:
        for resolution in STANDARD_VIDEO_RESOLUTIONS:
            path = videos / resolution / "mp4" / "shapes.mp4"
            cases.append(
                _case(
                    case_id=f"INGEST-VID-RES-{_transport_code(transport_mode)}-{resolution}",
                    category="视频分辨率读取",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=INGEST_PROMPT,
                    content=[
                        text_content(INGEST_PROMPT),
                        build_media_content(path, "video", transport_mode, media_base_url, project_root),
                    ],
                    files=[path],
                    resolution=resolution,
                    media_format="mp4",
                    max_completion_tokens=INGEST_MAX_TOKENS,
                )
            )

    return cases


def build_standard_semantic_cases(
    project_root: Path,
    media_base_url: str | None,
    transport_modes: Iterable[str],
    full_transport_matrix: bool = False,
) -> list[TestCase]:
    modes = normalize_transport_modes(transport_modes, media_base_url)
    default_modes = modes if full_transport_matrix else primary_transport_modes(modes)
    triple_modes = modes
    pics = project_root / "pics"
    videos = project_root / "video"
    suite_name = "phase2_semantic_standard"
    cases: list[TestCase] = []

    for transport_mode in triple_modes:
        for extension in IMAGE_FORMATS:
            shape_name = "rectangle" if transport_mode != "base64" else "triangle"
            path = pics / "720x1280" / extension / f"{shape_name}.{extension}"
            cases.append(
                _case(
                    case_id=f"IMG-{_transport_code(transport_mode)}-{extension.upper()}",
                    category="图片单图语义理解",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="image",
                    transport_mode=transport_mode,
                    prompt=PROMPT_DESCRIBE_IMAGE,
                    content=[
                        text_content(PROMPT_DESCRIBE_IMAGE),
                        build_media_content(path, "image", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=shape_expected_groups(shape_name),
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                    max_completion_tokens=512,
                )
            )

    for transport_mode in default_modes:
        for resolution in STANDARD_IMAGE_RESOLUTIONS:
            path = pics / resolution / "jpg" / "circle.jpg"
            cases.append(
                _case(
                    case_id=f"IMG-RES-{_transport_code(transport_mode)}-{resolution}",
                    category="图片分辨率语义理解",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="image",
                    transport_mode=transport_mode,
                    prompt=PROMPT_DESCRIBE_IMAGE,
                    content=[
                        text_content(PROMPT_DESCRIBE_IMAGE),
                        build_media_content(path, "image", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=shape_expected_groups("circle"),
                    files=[path],
                    resolution=resolution,
                    media_format="jpg",
                    max_completion_tokens=512,
                )
            )

    multi_paths = [pics / "720x1280" / "jpg" / f"{shape}.jpg" for shape in SHAPE_ORDER]
    for transport_mode in triple_modes:
        cases.append(
            _case(
                case_id=f"IMG-MULTI-7-{_transport_code(transport_mode)}",
                category="多图输入理解",
                suite_name=suite_name,
                media_scale="standard",
                media_type="image",
                transport_mode=transport_mode,
                prompt=PROMPT_MULTI_IMAGE,
                content=[
                    text_content(PROMPT_MULTI_IMAGE),
                    *[
                        build_media_content(path, "image", transport_mode, media_base_url, project_root)
                        for path in multi_paths
                    ],
                ],
                expected_groups=ordered_shape_expected_groups(),
                files=multi_paths,
                resolution="720x1280",
                media_format="jpg",
                max_completion_tokens=512,
                match_mode="ordered_groups",
            )
        )

    for transport_mode in triple_modes:
        first = pics / "720x1280" / "jpg" / "square.jpg"
        second = pics / "720x1280" / "jpg" / "circle.jpg"
        third = pics / "720x1280" / "jpg" / "triangle.jpg"
        cases.append(
            _case(
                case_id=f"IMG-INTERLEAVE-2-{_transport_code(transport_mode)}",
                category="图文穿插输入理解",
                suite_name=suite_name,
                media_scale="standard",
                media_type="image",
                transport_mode=transport_mode,
                prompt=PROMPT_INTERLEAVE_TWO,
                content=[
                    text_content("第一张图如下。"),
                    build_media_content(first, "image", transport_mode, media_base_url, project_root),
                    text_content("第二张图如下。"),
                    build_media_content(second, "image", transport_mode, media_base_url, project_root),
                    text_content(PROMPT_INTERLEAVE_TWO),
                ],
                expected_groups=[SHAPE_SYNONYMS["square"], SHAPE_SYNONYMS["circle"]],
                files=[first, second],
                resolution="720x1280",
                media_format="jpg",
                max_completion_tokens=512,
                match_mode="ordered_groups",
            )
        )
        cases.append(
            _case(
                case_id=f"IMG-INTERLEAVE-3-{_transport_code(transport_mode)}",
                category="图文穿插输入理解",
                suite_name=suite_name,
                media_scale="standard",
                media_type="image",
                transport_mode=transport_mode,
                prompt=PROMPT_INTERLEAVE_THREE,
                content=[
                    build_media_content(first, "image", transport_mode, media_base_url, project_root),
                    text_content("这是第一张。"),
                    build_media_content(second, "image", transport_mode, media_base_url, project_root),
                    text_content("这是第二张。"),
                    build_media_content(third, "image", transport_mode, media_base_url, project_root),
                    text_content(f"这是第三张。{PROMPT_INTERLEAVE_THREE}"),
                ],
                expected_groups=[SHAPE_SYNONYMS["square"], SHAPE_SYNONYMS["circle"], SHAPE_SYNONYMS["triangle"]],
                files=[first, second, third],
                resolution="720x1280",
                media_format="jpg",
                max_completion_tokens=512,
                match_mode="ordered_groups",
            )
        )

    for transport_mode in default_modes:
        for extension in VIDEO_FORMATS:
            path = videos / "720x1280" / extension / f"shapes.{extension}"
            cases.append(
                _case(
                    case_id=f"VID-{_transport_code(transport_mode)}-{extension.upper()}",
                    category="视频语义理解",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=PROMPT_VIDEO_ORDER,
                    content=[
                        text_content(PROMPT_VIDEO_ORDER),
                        build_media_content(path, "video", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=ordered_shape_expected_groups(),
                    files=[path],
                    resolution="720x1280",
                    media_format=extension,
                    max_completion_tokens=512,
                    match_mode="ordered_groups",
                )
            )

    for transport_mode in default_modes:
        for resolution in STANDARD_VIDEO_RESOLUTIONS:
            path = videos / resolution / "mp4" / "shapes.mp4"
            cases.append(
                _case(
                    case_id=f"VID-RES-{_transport_code(transport_mode)}-{resolution}",
                    category="视频分辨率语义理解",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=PROMPT_VIDEO_ORDER,
                    content=[
                        text_content(PROMPT_VIDEO_ORDER),
                        build_media_content(path, "video", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=ordered_shape_expected_groups(),
                    files=[path],
                    resolution=resolution,
                    media_format="mp4",
                    max_completion_tokens=512,
                    match_mode="ordered_groups",
                )
            )

    for transport_mode in default_modes:
        mp4_video = videos / "720x1280" / "mp4" / "shapes.mp4"
        cases.extend(
            [
                _case(
                    case_id=f"VID-FIRST-{_transport_code(transport_mode)}",
                    category="视频理解细节",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=PROMPT_VIDEO_FIRST,
                    content=[
                        text_content(PROMPT_VIDEO_FIRST),
                        build_media_content(mp4_video, "video", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=[SHAPE_SYNONYMS["square"]],
                    files=[mp4_video],
                    resolution="720x1280",
                    media_format="mp4",
                    max_completion_tokens=512,
                ),
                _case(
                    case_id=f"VID-LAST-{_transport_code(transport_mode)}",
                    category="视频理解细节",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=PROMPT_VIDEO_LAST,
                    content=[
                        text_content(PROMPT_VIDEO_LAST),
                        build_media_content(mp4_video, "video", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=[SHAPE_SYNONYMS["cube"]],
                    files=[mp4_video],
                    resolution="720x1280",
                    media_format="mp4",
                    max_completion_tokens=512,
                ),
                _case(
                    case_id=f"VID-ORDER-{_transport_code(transport_mode)}",
                    category="视频理解细节",
                    suite_name=suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=PROMPT_VIDEO_ORDER,
                    content=[
                        text_content(PROMPT_VIDEO_ORDER),
                        build_media_content(mp4_video, "video", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=ordered_shape_expected_groups(),
                    files=[mp4_video],
                    resolution="720x1280",
                    media_format="mp4",
                    max_completion_tokens=512,
                    match_mode="ordered_groups",
                ),
            ]
        )

    multi_video_suite_name = "phase2_semantic_multi_video_standard"
    for transport_mode in default_modes:
        for count in range(1, len(MULTI_VIDEO_SEQUENCE) + 1):
            shape_sequence = list(MULTI_VIDEO_SEQUENCE[:count])
            video_paths = [videos / "720x1280" / "mp4" / f"{shape}.mp4" for shape in shape_sequence]
            cases.append(
                _case(
                    case_id=f"VID-MULTI-{count:02d}-{_transport_code(transport_mode)}",
                    category="多视频输入理解",
                    suite_name=multi_video_suite_name,
                    media_scale="standard",
                    media_type="video",
                    transport_mode=transport_mode,
                    prompt=PROMPT_MULTI_VIDEO,
                    content=[
                        text_content(PROMPT_MULTI_VIDEO),
                        *[
                            build_media_content(path, "video", transport_mode, media_base_url, project_root)
                            for path in video_paths
                        ],
                    ],
                    expected_groups=[SHAPE_SYNONYMS[shape] for shape in shape_sequence],
                    files=video_paths,
                    resolution="720x1280",
                    media_format="mp4",
                    max_completion_tokens=512,
                    match_mode="ordered_groups",
                )
            )

    return cases


def build_large_image_ingestion_cases(
    project_root: Path,
    media_base_url: str | None,
    transport_modes: Iterable[str],
    resolutions: Iterable[str],
    full_transport_matrix: bool = False,
) -> list[TestCase]:
    modes = normalize_transport_modes(transport_modes, media_base_url)
    default_modes = modes if full_transport_matrix else primary_transport_modes(modes)
    pics = project_root / "pics"
    suite_name = "phase1_ingestion_large_image_smoke"
    cases: list[TestCase] = []
    for transport_mode in default_modes:
        for resolution in resolutions:
            path = pics / resolution / "jpg" / "square.jpg"
            cases.append(
                _case(
                    case_id=f"INGEST-IMG-LARGE-{_transport_code(transport_mode)}-{resolution}",
                    category="大图分辨率读取（Smoke）",
                    suite_name=suite_name,
                    media_scale="large",
                    media_type="image",
                    transport_mode=transport_mode,
                    prompt=INGEST_PROMPT,
                    content=[
                        text_content(INGEST_PROMPT),
                        build_media_content(path, "image", transport_mode, media_base_url, project_root),
                    ],
                    files=[path],
                    resolution=resolution,
                    media_format="jpg",
                    max_completion_tokens=INGEST_MAX_TOKENS,
                )
            )
    return cases


def build_large_image_semantic_cases(
    project_root: Path,
    media_base_url: str | None,
    transport_modes: Iterable[str],
    resolutions: Iterable[str],
    full_transport_matrix: bool = False,
) -> list[TestCase]:
    modes = normalize_transport_modes(transport_modes, media_base_url)
    default_modes = modes if full_transport_matrix else primary_transport_modes(modes)
    pics = project_root / "pics"
    suite_name = "phase2_semantic_large_image_smoke"
    cases: list[TestCase] = []
    for transport_mode in default_modes:
        for resolution in resolutions:
            path = pics / resolution / "jpg" / "square.jpg"
            cases.append(
                _case(
                    case_id=f"IMG-LARGE-{_transport_code(transport_mode)}-{resolution}",
                    category="大图分辨率理解（Smoke）",
                    suite_name=suite_name,
                    media_scale="large",
                    media_type="image",
                    transport_mode=transport_mode,
                    prompt=PROMPT_DESCRIBE_IMAGE,
                    content=[
                        text_content(PROMPT_DESCRIBE_IMAGE),
                        build_media_content(path, "image", transport_mode, media_base_url, project_root),
                    ],
                    expected_groups=shape_expected_groups("square"),
                    files=[path],
                    resolution=resolution,
                    media_format="jpg",
                    max_completion_tokens=512,
                )
            )
    return cases


def build_capability_cases(
    project_root: Path,
    media_base_url: str | None,
    transport_modes: Iterable[str],
    include_large_image_smoke: bool,
    large_image_resolutions: Iterable[str],
    full_transport_matrix: bool = False,
) -> tuple[list[TestCase], list[TestCase], list[str]]:
    normalized_modes = normalize_transport_modes(transport_modes, media_base_url)
    ingestion_cases = build_standard_ingestion_cases(
        project_root,
        media_base_url,
        normalized_modes,
        full_transport_matrix=full_transport_matrix,
    )
    semantic_cases = build_standard_semantic_cases(
        project_root,
        media_base_url,
        normalized_modes,
        full_transport_matrix=full_transport_matrix,
    )
    enabled_optional_suites: list[str] = []
    if include_large_image_smoke:
        enabled_optional_suites.append("large_image_smoke")
        ingestion_cases.extend(
            build_large_image_ingestion_cases(
                project_root,
                media_base_url,
                normalized_modes,
                large_image_resolutions,
                full_transport_matrix=full_transport_matrix,
            )
        )
        semantic_cases.extend(
            build_large_image_semantic_cases(
                project_root,
                media_base_url,
                normalized_modes,
                large_image_resolutions,
                full_transport_matrix=full_transport_matrix,
            )
        )
    return ingestion_cases, semantic_cases, enabled_optional_suites


def assert_case_files(cases: Iterable[TestCase]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for case in cases:
        missing = [str(path) for path in case.files if not path.exists()]
        if missing:
            issues.append({"case_id": case.case_id, "missing_files": missing})
    return issues
