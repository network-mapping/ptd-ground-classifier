"""Run manifest: which version produced the output, with which parameters.

One manifest.json is written into the output folder at the end of every run and
overwrites the previous one. Each output LAS also carries the version in its
`generating_software` header field, which stays with the file.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict | None:
    """Commit and dirty flag of the checkout this code runs from; None if not a git checkout."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions() -> dict:
    import laspy
    import numpy
    import scipy

    from . import __version__, ptdcore

    return {
        "name": "ground-classifier",
        "version": __version__,
        "core_version": ptdcore.CORE_VERSION,
        "git": git_state(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "laspy": laspy.__version__,
    }


def machine_info() -> dict:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def file_record(
    input_path: Path,
    output_path: Path,
    status: str = "not_processed",
    unit: str | None = None,
) -> dict:
    stat = input_path.stat()
    return {
        "input": str(input_path),
        "input_size_bytes": stat.st_size,
        "input_mtime_utc": utc_from_timestamp(stat.st_mtime),
        "unit": unit,
        "output": str(output_path),
        "status": status,
    }


def write_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def build_manifest(
    *,
    started: str,
    finished: str,
    duration_s: float,
    status: str,
    error: str | None,
    cli: dict,
    config: dict,
    files: list[dict],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "software": software_versions(),
        "run": {
            "status": status,
            "error": error,
            "started_utc": started,
            "finished_utc": finished,
            "duration_s": round(duration_s, 1),
            "command": sys.argv,
            **machine_info(),
        },
        "parameters": {"cli": cli, "config": config},
        "files": files,
    }
