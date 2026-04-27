#!/usr/bin/env python3
"""Shared helpers for building multimodal media references."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import quote


def add_media_mode_args(parser):
    parser.add_argument(
        "--media-mode",
        choices=["base64", "http", "local_path"],
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
        help=(
            "Local media root used for existing image files or for materialized images "
            "when media-mode is http or local_path."
        ),
    )


def _sanitize_relative_path(image_path: str | None, fallback_name: str) -> Path:
    if image_path:
        raw = Path(str(image_path))
        if raw.is_absolute():
            return Path(raw.name)
        return Path(*[part for part in raw.parts if part not in {"", ".", ".."}])
    return Path(fallback_name)


def _guess_suffix(image_path: str | None, default_suffix: str) -> str:
    if image_path:
        suffix = Path(str(image_path)).suffix.lower()
        if suffix:
            return suffix
    return default_suffix


def _materialize_binary(
    encoded_payload: str,
    media_root: Path,
    relative_path: Path,
) -> Path:
    target = media_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(base64.b64decode(encoded_payload))
    return target.resolve()


def _path_to_file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _path_to_http_url(path: Path, media_root: Path, media_base_url: str) -> str:
    resolved_path = path.resolve()
    resolved_root = media_root.resolve()
    relative = resolved_path.relative_to(resolved_root)
    quoted = "/".join(quote(part) for part in relative.parts)
    return media_base_url.rstrip("/") + "/" + quoted


def _guess_mime_type(path: Path, fallback_mime: str) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or fallback_mime


def build_media_reference(
    *,
    encoded_payload: str | None,
    source_path: str | None,
    media_mode: str,
    media_root: str,
    media_base_url: str,
    fallback_name: str,
    fallback_suffix: str,
    fallback_mime: str,
) -> tuple[str, str]:
    """Return the media reference plus the resolved local path when available."""
    if media_mode == "base64":
        if source_path and Path(source_path).expanduser().exists():
            resolved = Path(source_path).expanduser().resolve()
            payload = base64.b64encode(resolved.read_bytes()).decode("ascii")
            mime_type = _guess_mime_type(resolved, fallback_mime)
            return f"data:{mime_type};base64,{payload}", str(resolved)
        if not encoded_payload:
            raise ValueError("base64 mode requires encoded_payload or an existing source path")
        return f"data:{fallback_mime};base64,{encoded_payload}", ""

    relative_path = _sanitize_relative_path(source_path, fallback_name)
    guessed_suffix = _guess_suffix(source_path, fallback_suffix)
    if not relative_path.suffix:
        relative_path = relative_path.with_suffix(guessed_suffix)

    root = Path(media_root).expanduser() if media_root else None

    resolved_path: Path | None = None
    if source_path:
        candidate = Path(str(source_path)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            resolved_path = candidate.resolve()
        elif root:
            rooted = (root / candidate).resolve()
            if rooted.exists():
                resolved_path = rooted

    if resolved_path is None:
        if not encoded_payload:
            raise ValueError(
                f"{media_mode} mode requires either an existing local path or encoded payload for {fallback_name}"
            )
        if root is None:
            raise ValueError(f"{media_mode} mode requires --media-root when local files are not already present")
        resolved_path = _materialize_binary(encoded_payload, root, relative_path)

    if media_mode == "local_path":
        return _path_to_file_url(resolved_path), str(resolved_path)

    if not media_base_url:
        raise ValueError("http mode requires --media-base-url")
    if root is None:
        raise ValueError("http mode requires --media-root")
    return _path_to_http_url(resolved_path, root, media_base_url), str(resolved_path)


def build_image_reference(
    *,
    image_b64: str | None,
    image_path: str | None,
    media_mode: str,
    media_root: str,
    media_base_url: str,
    fallback_name: str,
) -> tuple[str, str]:
    return build_media_reference(
        encoded_payload=image_b64,
        source_path=image_path,
        media_mode=media_mode,
        media_root=media_root,
        media_base_url=media_base_url,
        fallback_name=fallback_name,
        fallback_suffix=".jpg",
        fallback_mime="image/jpeg",
    )
