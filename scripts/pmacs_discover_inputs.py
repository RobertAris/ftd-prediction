#!/usr/bin/env python3
"""Bounded metadata discovery; only explicitly selected source scripts are read."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import stat


# Match inspect_formats.py's depth semantics and inventory ceilings.
DEFAULTS = dict(max_depth=3, max_entries=10_000, max_candidates=200,
                max_source_bytes=262_144, max_source_lines=80, max_line_length=240)
HARD_LIMITS = dict(max_depth=16, max_entries=100_000, max_candidates=2_000,
                   max_source_bytes=1_048_576, max_source_lines=400, max_line_length=1_000)
MAX_ROOTS = 16
MAX_SOURCES = 16
PRUNE_NAMES = frozenset({
    "raw", "raw_data", "rawdata", "logs", "temp_seg_inputs", "outs", "tmp", "temp",
    ".git", ".venv", "venv", "03_segmentation",
})
PRUNE_TOKENS = ("fastq", "bam", "cellranger", "cellbender", "scrublet",
                "intermediate", "cache", "model", "checkpoint")
ASSAY_SUFFIXES = (".h5ad", ".h5seurat", ".rds", ".rda", ".h5", ".hdf5",
                  ".loom", ".mtx", ".mtx.gz", ".zarr")
RECORD_PATTERN = re.compile(r"metadata|mapping|manifest|sample|donor|section|readme|provenance", re.I)
SOURCE_PATTERN = re.compile(r"read|load|save|write|raw|layers?|normali[sz]|log1p|section|donor|cell[_. -]?type|counts?", re.I)
PLACEHOLDER = re.compile(r"[<>{}$]|placeholder|replace[_-]?me|\b(?:tbd|todo)\b|your[_-]", re.I)
AUTHORITY = [
    "All candidates and source matches are unverified; filenames are not diagnoses.",
    "Filesystem ownership and timestamps do not identify the designated scientific owner.",
    "Discovery cannot establish donor identity, physical sections, canonical outputs or count provenance.",
    "Keep unpaired donor cohorts separate; verify independent donor replication and physical-section mappings.",
    "No assay/table contents read, script execution, model raw-count probe or analysis authorization.",
]


class Refusal(Exception):
    def __init__(self, code):
        self.code = code


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's ordinary diagnostic may contain an input path or identifier.
        raise Refusal("invalid_arguments")


def supported():
    return (hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
            and os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
            and os.scandir in os.supports_fd)


def directory_flags():
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def open_directory(root_fd, parts):
    """Reopen from a pinned handle, refusing links even after enumeration races."""
    current = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(part, directory_flags(), dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def open_absolute_directory(path):
    anchor = os.open(path.anchor, directory_flags())
    try:
        return open_directory(anchor, path.parts[1:])
    finally:
        os.close(anchor)


def selected_path(value):
    path = Path(value)
    if (not value.strip() or not path.is_absolute() or ".." in path.parts
            or PLACEHOLDER.search(value)):
        raise Refusal("invalid_path")
    return path


def pruned(name):
    name = name.lower()
    return name in PRUNE_NAMES or any(token in name for token in PRUNE_TOKENS)


def within(path, root):
    return path == root or root in path.parents


def mark_truncated(report, reason):
    if reason not in report["truncation_reasons"]:
        report["truncation_reasons"].append(reason)


def record_error(report, code, error, path=None):
    report["counts"]["errors"] += 1
    if code not in report["reason_codes"]:
        report["reason_codes"].append(code)
    if len(report["private_errors"]) < 100:
        report["private_errors"].append({"code": code, "path": str(path) if path else None,
                                         "detail": str(error)[:1_000]})
    else:
        report["counts"]["error_details_omitted"] += 1


def add_candidate(report, path, kind, info):
    report["candidates"].append({"path": str(path), "kind": kind, "verified": False,
                                  "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
    if len(report["candidates"]) == report["limits"]["max_candidates"]:
        mark_truncated(report, "max_candidates")


def candidate_kind(name):
    lowered = name.lower()
    if lowered.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".sam", ".cram")):
        return None
    if lowered.endswith((".py", ".r")):
        return "preprocessing_script_candidate"
    if lowered.endswith(ASSAY_SUFFIXES):
        return "assay_container_candidate"
    if RECORD_PATTERN.search(name):
        return "mapping_or_metadata_candidate"
    return None


def scan_root(report, root_fd, canonical):
    limits, counts = report["limits"], report["counts"]
    if canonical.name.lower().endswith(".zarr"):
        add_candidate(report, canonical, "atomic_zarr_candidate", os.fstat(root_fd))
        counts["atomic_zarr"] += 1
        return
    pending = [((), 0)]
    while pending and not any(x in report["truncation_reasons"] for x in ("max_entries", "max_candidates")):
        parts, depth = pending.pop()
        directory = canonical.joinpath(*parts)
        try:
            directory_fd = open_directory(root_fd, parts)
            try:
                with os.scandir(directory_fd) as entries:
                    counts["directories_scanned"] += 1
                    for entry in entries:
                        counts["entries_examined"] += 1
                        path = directory / entry.name
                        try:
                            info = entry.stat(follow_symlinks=False)
                            if stat.S_ISLNK(info.st_mode):
                                counts["symlinks_skipped"] += 1
                            elif stat.S_ISDIR(info.st_mode):
                                if entry.name.lower().endswith(".zarr"):
                                    add_candidate(report, path, "atomic_zarr_candidate", info)
                                    counts["atomic_zarr"] += 1
                                elif pruned(entry.name):
                                    counts["directories_pruned"] += 1
                                elif depth < limits["max_depth"]:
                                    pending.append(((*parts, entry.name), depth + 1))
                                else:
                                    counts["directories_skipped_depth"] += 1
                                    mark_truncated(report, "max_depth")
                            elif stat.S_ISREG(info.st_mode):
                                kind = candidate_kind(entry.name)
                                if kind:
                                    add_candidate(report, path, kind, info)
                            else:
                                counts["other_entries_skipped"] += 1
                        except OSError as error:
                            record_error(report, "entry_error", error, path)
                        if counts["entries_examined"] == limits["max_entries"]:
                            mark_truncated(report, "max_entries")
                        if any(x in report["truncation_reasons"] for x in ("max_entries", "max_candidates")):
                            break
            finally:
                os.close(directory_fd)
        except OSError as error:
            record_error(report, "directory_error", error, directory)


def source_location(value, roots):
    path = selected_path(value)
    if path.suffix.lower() not in {".py", ".r"}:
        raise Refusal("source_extension_refused")
    # No resolve() on a source: an intermediate or final symlink is forbidden.
    for supplied, canonical, root_fd in roots:
        for base in (supplied, canonical):
            if within(path, base) and path != base:
                parts = path.relative_to(base).parts
                if canonical.name.lower().endswith(".zarr") or any(
                        pruned(p) or p.lower().endswith(".zarr") for p in parts[:-1]):
                    raise Refusal("source_pruned")
                return canonical.joinpath(*parts), root_fd, parts
    raise Refusal("source_outside_roots")


def extract_source(report, value, roots):
    record = {"selected_path": value, "path": None, "status": "incomplete", "bytes_read": 0,
              "matches_in_prefix": 0, "lines_clipped": 0, "limitations": [], "snippets": []}
    report["sources"].append(record)
    limits = report["limits"]
    try:
        path, root_fd, parts = source_location(value, roots)
        record["path"] = str(path)
        if report["counts"]["source_lines_emitted"] >= limits["max_source_lines"]:
            record["limitations"].append("max_source_lines")
            mark_truncated(report, "max_source_lines")
            return
        parent_fd = open_directory(root_fd, parts[:-1])
        try:
            before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise Refusal("source_not_regular_nonsymlink")
            source_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                dir_fd=parent_fd)
            try:
                initial = os.fstat(source_fd)
                if (not stat.S_ISREG(initial.st_mode)
                        or (before.st_dev, before.st_ino) != (initial.st_dev, initial.st_ino)):
                    raise Refusal("source_changed")
                data = bytearray()
                while len(data) < limits["max_source_bytes"]:
                    chunk = os.read(source_fd, min(65_536, limits["max_source_bytes"] - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    record["bytes_read"] += len(chunk)
                    report["counts"]["source_bytes_read"] += len(chunk)
                after = os.fstat(source_fd)
            finally:
                os.close(source_fd)
        finally:
            os.close(parent_fd)
        if (initial.st_size, initial.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            record["limitations"].append("source_changed")
            record_error(report, "source_changed", Refusal("source_changed"), path)
        byte_cut = max(initial.st_size, after.st_size) > len(data)
        if byte_cut:
            record["limitations"].append("max_source_bytes")
            mark_truncated(report, "max_source_bytes")
        text = data.decode("utf-8", errors="replace")
        if "\ufffd" in text:
            record["limitations"].append("utf8_replacement")
        lines = text.splitlines()
        if byte_cut and lines and not text.endswith(("\n", "\r")):
            lines.pop()
            record["limitations"].append("partial_last_line_omitted")
        for number, line in enumerate(lines, 1):
            if not SOURCE_PATTERN.search(line):
                continue
            record["matches_in_prefix"] += 1
            if report["counts"]["source_lines_emitted"] >= limits["max_source_lines"]:
                if "max_source_lines" not in record["limitations"]:
                    record["limitations"].append("max_source_lines")
                mark_truncated(report, "max_source_lines")
                continue
            clipped = len(line) > limits["max_line_length"]
            if clipped:
                record["lines_clipped"] += 1
                if "max_line_length" not in record["limitations"]:
                    record["limitations"].append("max_line_length")
                mark_truncated(report, "max_line_length")
            record["snippets"].append({"line": number, "text": line[:limits["max_line_length"]],
                                       "clipped": clipped})
            report["counts"]["source_lines_emitted"] += 1
        record["status"] = "incomplete" if record["limitations"] else "ok"
    except (OSError, ValueError, Refusal) as error:
        record["status"] = "error"
        code = error.code if isinstance(error, Refusal) else "source_unavailable"
        record["limitations"].append(code)
        record_error(report, code, error, value)


def create_output(value, roots):
    supplied = selected_path(value)
    parent = supplied.parent.resolve(strict=True)
    canonical = parent / supplied.name
    if any(within(canonical, root) for _, root, _ in roots):
        raise Refusal("output_inside_root")
    parent_fd = open_absolute_directory(parent)
    try:
        # mkdir is exclusive even for broken final symlinks; no overwrite/fallback.
        os.mkdir(supplied.name, mode=0o700, dir_fd=parent_fd)
        output_fd = os.open(supplied.name, directory_flags(), dir_fd=parent_fd)
        return canonical, output_fd
    finally:
        os.close(parent_fd)


def private_write(output_fd, name, text):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=output_fd)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def summary(report):
    return {"status": report["status"], "complete": report["status"] == "complete",
            "counts": report["counts"], "reason_codes": report["reason_codes"],
            "truncation_reasons": report["truncation_reasons"]}


def discover(args):
    limits = {name: getattr(args, name) for name in DEFAULTS}
    for name, value in limits.items():
        if not (0 if name == "max_depth" else 1) <= value <= HARD_LIMITS[name]:
            raise Refusal("invalid_limits")
    if len(args.root) > MAX_ROOTS or len(args.source_file) > MAX_SOURCES:
        raise Refusal("too_many_selections")
    if not supported():
        raise Refusal("unsupported_platform")
    report = {"schema_version": 1, "status": "incomplete", "limits": limits,
              "hard_limits": {**HARD_LIMITS, "max_roots": MAX_ROOTS, "max_sources": MAX_SOURCES},
              "authority_limits": AUTHORITY, "roots": [], "candidates": [], "sources": [],
              "reason_codes": [], "truncation_reasons": [], "private_errors": [],
              "counts": dict.fromkeys(("entries_examined", "directories_scanned", "directories_pruned",
                                       "directories_skipped_depth", "symlinks_skipped", "atomic_zarr",
                                       "other_entries_skipped", "source_bytes_read", "source_lines_emitted",
                                       "errors", "error_details_omitted", "roots_scanned"), 0)}
    with ExitStack() as stack:
        roots = []
        for value in args.root:
            supplied = selected_path(value)
            canonical = supplied.resolve(strict=True)
            # At least three components below /, checked both before and after resolution.
            if len(supplied.parts) < 4 or len(canonical.parts) < 4:
                raise Refusal("root_too_broad")
            if (pruned(canonical.name) and canonical.name.lower() != "03_segmentation"
                    and not canonical.name.lower().endswith(".zarr")):
                raise Refusal("root_pruned")
            root_fd = open_absolute_directory(canonical)
            stack.callback(os.close, root_fd)
            roots.append((supplied, canonical, root_fd))
            report["roots"].append({"selected_path": str(supplied), "canonical_path": str(canonical)})
        output, output_fd = create_output(args.output, roots)
        stack.callback(os.close, output_fd)
        report["output_directory"] = str(output)
        try:
            for _, canonical, root_fd in roots:
                if any(x in report["truncation_reasons"] for x in ("max_entries", "max_candidates")):
                    break
                scan_root(report, root_fd, canonical)
                report["counts"]["roots_scanned"] += 1
            for value in args.source_file:
                extract_source(report, value, roots)
        except (Exception, KeyboardInterrupt) as error:
            record_error(report, "discovery_failed", error)
        report["counts"]["candidates"] = len(report["candidates"])
        report["counts"]["sources_selected"] = len(args.source_file)
        # JSON-escaped text records prevent control characters in paths/source from becoming terminal controls.
        snippets = [json.dumps({"path": source["path"], **snippet})
                    for source in report["sources"] for snippet in source["snippets"]]
        try:
            private_write(output_fd, "snippets.txt", "\n".join(snippets) + ("\n" if snippets else ""))
        except OSError as error:
            record_error(report, "snippet_write_error", error)
        report["status"] = ("incomplete" if report["truncation_reasons"] or report["counts"]["errors"]
                            or any(s["limitations"] for s in report["sources"]) else "complete")
        try:
            private_write(output_fd, "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
        except OSError as error:
            record_error(report, "report_write_error", error)
            report["status"] = "incomplete"
            try:
                private_write(output_fd, "failure.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
            except OSError:
                pass  # A full/unwritable filesystem may prevent even private diagnostics.
    return report


def main(argv=None):
    parser = Parser(prog="pmacs-discover-inputs", allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="Exact approved stage/directory (repeatable).")
    parser.add_argument("--output", required=True, help="New private directory outside every root; parent must exist.")
    parser.add_argument("--source-file", action="append", default=[], help="Exact selected .py/.R inside roots (repeatable).")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default,
                            help=f"Default {default}; hard maximum {HARD_LIMITS[name]}.")
    previous_umask = os.umask(0o077)
    try:
        args = parser.parse_args(argv)
        try:
            report = discover(args)
        except Refusal:
            raise
        except OSError:
            raise Refusal("filesystem_refused") from None
        print(json.dumps(summary(report), sort_keys=True))
        return 1 if report["counts"]["errors"] else (2 if report["status"] == "incomplete" else 0)
    except (Exception, KeyboardInterrupt) as error:
        code = error.code if isinstance(error, Refusal) else "discovery_failed"
        print(json.dumps({"status": "error", "complete": False, "reason_codes": [code]}))
        return 1
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
