#!/usr/bin/env python3
"""Bounded, read-only directory-entry inventory with aggregate output only.

Assay files are never opened or read. Installed distribution metadata is queried
only through importlib.metadata; computational/GPU packages are not imported.
Run against one explicitly approved PMACS assay-output directory.
"""

from __future__ import annotations

import argparse
import errno
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys


DEFAULT_MAX_DEPTH = 3
HARD_MAX_DEPTH = 16
DEFAULT_MAX_ENTRIES = 10_000
HARD_MAX_ENTRIES = 100_000
PACKAGES = ("numpy", "scipy", "anndata", "h5py", "scanpy", "scvi-tools", "novae", "torch")
FILE_FORMATS = (
    ".mtx.gz", ".csv.gz", ".tsv.gz", ".h5seurat", ".parquet", ".h5ad",
    ".loom", ".rds", ".rda", ".mtx", ".csv", ".tsv", ".h5",
)


def _payload(max_depth: int, max_entries: int) -> dict:
    return {
        "schema_version": 1,
        "status": "incomplete",
        "complete": False,
        "truncated": False,
        "truncation_reasons": [],
        "reason_codes": [],
        "limits": {"max_depth": max_depth, "hard_max_depth": HARD_MAX_DEPTH,
                   "max_entries": max_entries, "hard_max_entries": HARD_MAX_ENTRIES},
        "depth_definition": "Maximum descendant directory level scanned; root files are included at depth zero.",
        "entries_examined": 0,
        "regular_files": 0,
        "directory_entries": 0,
        "directories_scanned": 0,
        "directories_skipped_depth": 0,
        "symlinks_skipped": 0,
        "other_entries": 0,
        "error_count": 0,
        "permission_error_count": 0,
        "version_lookup_error_count": 0,
        "format_counts": {**{suffix[1:]: 0 for suffix in FILE_FORMATS}, "zarr": 0, "unknown": 0},
        "python_version": platform.python_version(),
        "installed_versions": {name: None for name in PACKAGES},
        "scope": "Directory-entry metadata only; zarr directories are atomic containers and child symlinks are excluded.",
    }


def _reason(result: dict, code: str) -> None:
    if code not in result["reason_codes"]:
        result["reason_codes"].append(code)


def _error(result: dict, error: BaseException, code: str) -> None:
    result["error_count"] += 1
    if isinstance(error, PermissionError) or getattr(error, "errno", None) in {errno.EACCES, errno.EPERM}:
        result["permission_error_count"] += 1
    _reason(result, code)


def _truncate(result: dict, reason: str) -> None:
    result["truncated"] = True
    if reason not in result["truncation_reasons"]:
        result["truncation_reasons"].append(reason)


def _versions(result: dict) -> None:
    for name in PACKAGES:
        try:
            result["installed_versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # An absent optional package is a measured environment fact.
            pass
        except Exception as error:
            result["version_lookup_error_count"] += 1
            _error(result, error, "package_metadata_error")


def _supported() -> bool:
    return (
        hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd and os.scandir in os.supports_fd
    )


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_relative_directory(root_fd: int, parts: tuple[str, ...]) -> int:
    """Reopen only directory components below a fixed root, never child links."""
    current = os.dup(root_fd)
    try:
        for component in parts:
            try:
                child = os.open(component, _directory_flags(), dir_fd=current)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    try:
                        entry_stat = os.stat(component, dir_fd=current, follow_symlinks=False)
                    except OSError:
                        pass
                    else:
                        if stat.S_ISLNK(entry_stat.st_mode):
                            raise OSError(errno.ELOOP, "child symlink excluded") from None
                raise
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _file_format(name: str) -> str:
    lowered = name.lower()
    return next((suffix[1:] for suffix in FILE_FORMATS if lowered.endswith(suffix)), "unknown")


def inventory(root: str | Path, *, max_depth: int = DEFAULT_MAX_DEPTH,
              max_entries: int = DEFAULT_MAX_ENTRIES) -> dict:
    """Inspect no file contents; return no source path, name or error message.

    Directory handles with O_NOFOLLOW also prevent traversal if a queued child
    directory is replaced by a symlink after enumeration. Root links may resolve
    once to an existing canonical directory. No parent-directory fallback exists.
    """
    result = _payload(max_depth, max_entries)
    _versions(result)
    if type(max_depth) is not int or not 0 <= max_depth <= HARD_MAX_DEPTH or type(max_entries) is not int or not 1 <= max_entries <= HARD_MAX_ENTRIES:
        _error(result, ValueError(), "invalid_limits")
        return result
    if not isinstance(root, (str, Path)) or not str(root).strip():
        _error(result, ValueError(), "root_unavailable")
        return result
    if not _supported():
        _error(result, RuntimeError(), "unsupported_platform")
        return result
    try:
        supplied = Path(root)
        canonical = supplied.resolve(strict=True)
        root_fd = os.open(canonical, _directory_flags())
    except (OSError, RuntimeError, ValueError) as error:
        _error(result, error, "root_unavailable")
        return result

    try:
        if canonical.name.lower().endswith(".zarr") or supplied.name.lower().endswith(".zarr"):
            result["format_counts"]["zarr"] += 1
        else:
            pending: list[tuple[tuple[str, ...], int]] = [((), 0)]
            while pending and "max_entries" not in result["truncation_reasons"]:
                parts, depth = pending.pop()
                try:
                    directory_fd = _open_relative_directory(root_fd, parts)
                except OSError as error:
                    if error.errno == errno.ELOOP:
                        result["symlinks_skipped"] += 1
                    _error(result, error, "directory_error")
                    continue
                try:
                    with os.scandir(directory_fd) as entries:
                        result["directories_scanned"] += 1
                        for entry in entries:
                            if result["entries_examined"] >= max_entries:
                                _truncate(result, "max_entries")
                                break
                            result["entries_examined"] += 1
                            try:
                                if entry.is_symlink():
                                    result["symlinks_skipped"] += 1
                                elif entry.is_dir(follow_symlinks=False):
                                    result["directory_entries"] += 1
                                    if entry.name.lower().endswith(".zarr"):
                                        result["format_counts"]["zarr"] += 1
                                    elif depth < max_depth:
                                        pending.append(((*parts, entry.name), depth + 1))
                                    else:
                                        result["directories_skipped_depth"] += 1
                                        _truncate(result, "max_depth")
                                elif entry.is_file(follow_symlinks=False):
                                    result["regular_files"] += 1
                                    result["format_counts"][_file_format(entry.name)] += 1
                                else:
                                    result["other_entries"] += 1
                            except OSError as error:
                                _error(result, error, "entry_error")
                except OSError as error:
                    _error(result, error, "directory_error")
                finally:
                    os.close(directory_fd)
    finally:
        os.close(root_fd)

    result["complete"] = not result["truncated"] and result["error_count"] == 0
    result["status"] = "complete" if result["complete"] else "incomplete"
    return result


class _ArgumentError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _ArgumentError()


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog="inspect-formats", allow_abbrev=False,
                     description="Read-only, bounded format counts for one explicitly supplied assay-output directory.")
    parser.add_argument("--root", required=True, help="One approved assay-output directory; never printed.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Descendant directory levels to scan (default 3; hard maximum 16); root files are included at zero.")
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES,
                        help="Global entry limit (default 10000; hard maximum 100000).")
    try:
        args = parser.parse_args(argv)
        result = inventory(args.root, max_depth=args.max_depth, max_entries=args.max_entries)
    except _ArgumentError:
        result = _payload(DEFAULT_MAX_DEPTH, DEFAULT_MAX_ENTRIES)
        _versions(result)
        _error(result, ValueError(), "invalid_arguments")
    except Exception as error:
        # No exception repr, traceback, source path, filename or environment dump.
        result = _payload(DEFAULT_MAX_DEPTH, DEFAULT_MAX_ENTRIES)
        _error(result, error, "inventory_error")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["error_count"]:
        return 1
    return 2 if result["truncated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
