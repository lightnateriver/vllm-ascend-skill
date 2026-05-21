#!/usr/bin/env python3
"""Helpers for launching child Python scripts with optional RTK wrapping.

Callers should never assume RTK is installed. The default path is plain Python,
and RTK is enabled only after an executable is discovered via `RTK_BIN` or
`PATH`.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def resolve_rtk_executable() -> str | None:
    """Return an executable RTK path when available, else None.

    Resolution order:
    1. `RTK_BIN` environment variable, when it points to an executable path or
       a command name found on `PATH`.
    2. `rtk` discovered on `PATH`.
    """

    configured = os.environ.get("RTK_BIN", "").strip()
    if configured:
        if os.path.isabs(configured):
            return configured if os.access(configured, os.X_OK) else None
        discovered = shutil.which(configured)
        return discovered or None
    return shutil.which("rtk")


def build_python_cmd(script_path: str | Path, *args: str) -> list[str]:
    """Build a child Python command, enabling RTK only after a successful probe."""

    cmd = [sys.executable, str(script_path), *args]
    rtk_executable = resolve_rtk_executable()
    return [rtk_executable, *cmd] if rtk_executable else cmd


def build_child_env() -> dict[str, str]:
    """Build a stable child-process environment."""

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def build_python_launch(script_path: str | Path, *args: str) -> dict[str, Any]:
    """Return a launch spec with command, environment, and RTK probe metadata."""

    rtk_executable = resolve_rtk_executable()
    cmd = [sys.executable, str(script_path), *args]
    if rtk_executable:
        cmd = [rtk_executable, *cmd]
    env = build_child_env()
    return {
        "cmd": cmd,
        "env": env,
        "meta": {
            "used_rtk": bool(rtk_executable),
            "rtk_executable": rtk_executable or "",
            "python_executable": sys.executable,
            "path_head": env.get("PATH", "").split(os.pathsep)[:8],
        },
    }
