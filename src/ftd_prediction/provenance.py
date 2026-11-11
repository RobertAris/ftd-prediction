"""Reproducible, private run records without collecting credentials."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert NumPy objects without emitting invalid JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_json(path: str | Path, payload: Any, *, exclusive: bool = True) -> None:
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_snapshot(root: str | Path | None = None) -> dict:
    """Hash actual package sources, including uncommitted implementation files."""
    base = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    if (base / "src").exists():
        candidates = sorted((base / "src").rglob("*.py"))
    elif root is None:
        # Wheel installs have no repository src/ tree. Still fingerprint the
        # executing package rather than reporting the digest of an empty set.
        candidates = sorted(Path(__file__).resolve().parent.rglob("*.py"))
    else:
        candidates = []
    if (base / "pyproject.toml").is_file():
        candidates.append(base / "pyproject.toml")
    digest = hashlib.sha256()
    files = []
    for path in sorted(candidates):
        relative = path.relative_to(base).as_posix()
        checksum = file_sha256(path)
        digest.update(relative.encode("utf-8") + b"\0" + checksum.encode("ascii") + b"\n")
        files.append({"path": relative, "sha256": checksum})

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(base), *args], capture_output=True, text=True,
                check=False, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    tracked_text = git("ls-files", "--", "src", "pyproject.toml")
    tracked = set(tracked_text.splitlines()) if tracked_text else set()
    return {
        "git_revision": revision,
        "git_dirty": None if status is None else bool(status),
        "source_sha256": digest.hexdigest(),
        "source_files": files,
        "revision_covers_sources": bool(revision and status == "" and files and all(item["path"] in tracked for item in files)),
    }


def environment_record() -> dict:
    try:
        numpy_version = version("numpy")
    except PackageNotFoundError:
        numpy_version = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy_version,
    }


def input_record(paths: list[Path] | tuple[Path, ...]) -> list[dict]:
    """Paths and hashes are private run artifacts, not a sharing approval."""
    return [
        {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(set(Path(item).resolve() for item in paths))
    ]
