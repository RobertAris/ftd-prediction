"""Render a reviewed LSF request without connecting to or submitting a job."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path


class ClusterConfigError(ValueError):
    pass


_PLACEHOLDER = re.compile(r"replace|placeholder|<|>|your[_ -]|example", re.IGNORECASE)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
_RESOURCE = re.compile(r"^[A-Za-z0-9_\[\]=:.,+*/()!&| -]+$")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ClusterConfigError(f"{field} must be a nonempty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or _PLACEHOLDER.search(value):
        raise ClusterConfigError(f"{field} contains unresolved or unsafe content")
    return value


def _path(value: object, field: str) -> Path:
    text = _text(value, field)
    if not _PATH.fullmatch(text) or ".." in Path(text).parts:
        raise ClusterConfigError(f"{field} must be a plain absolute cluster path")
    return Path(text)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def render_lsf_text(config: dict) -> str:
    """Require explicit memory semantics; no guessed queue/GPU configuration."""
    if not isinstance(config, dict):
        raise ClusterConfigError("cluster configuration must be an object")
    workspace = _path(config.get("workspace"), "workspace")
    code = _path(config.get("code_directory"), "code_directory")
    python = _path(config.get("python_executable"), "python_executable")
    run = _path(config.get("run_directory"), "run_directory")
    if workspace == Path("/") or not _within(code, workspace) or not _within(run, workspace):
        raise ClusterConfigError("code and run directories must be within the personal workspace")
    if code == workspace or run == workspace or code == run or _within(run, code):
        raise ClusterConfigError("code and run directories must be distinct personal subdirectories")
    queue = _text(config.get("queue"), "queue")
    job = _text(config.get("job_name", "ftd_preflight"), "job_name")
    if not _IDENTIFIER.fullmatch(queue) or not _IDENTIFIER.fullmatch(job):
        raise ClusterConfigError("queue and job_name must be simple identifiers")
    cpus = config.get("cpu_slots")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ClusterConfigError("cpu_slots must be a positive integer")
    wall_time = _text(config.get("wall_time"), "wall_time")
    if not re.fullmatch(r"(?:[0-9]{1,3}):[0-5][0-9]", wall_time) or wall_time in {"0:00", "00:00", "000:00"}:
        raise ClusterConfigError("wall_time must be a positive HHH:MM limit")
    if config.get("memory_semantics_verified") is not True:
        raise ClusterConfigError("memory semantics must be verified before rendering")
    if config.get("run_directory_precreated") is not True:
        raise ClusterConfigError("confirm that the unique cluster run directory exists before rendering")
    if config.get("memory_scope") not in {"per_job", "per_slot"}:
        raise ClusterConfigError("memory_scope must be per_job or per_slot")
    if config.get("memory_units") not in {"MB", "MiB", "GB", "GiB", "KB", "KiB"}:
        raise ClusterConfigError("explicit verified memory_units are required")
    resource = _text(config.get("resource_expression"), "resource_expression")
    if not _RESOURCE.fullmatch(resource) or "rusage[" not in resource or "mem=" not in resource:
        raise ClusterConfigError("resource_expression must contain an explicit LSF memory reservation")
    for left, right in (("[", "]"), ("(", ")")):
        depth = 0
        for char in resource:
            depth += char == left
            depth -= char == right
            if depth < 0:
                raise ClusterConfigError("resource_expression is unbalanced")
        if depth:
            raise ClusterConfigError("resource_expression is unbalanced")
    arguments = config.get("arguments")
    if not isinstance(arguments, list) or not arguments or not all(isinstance(item, str) for item in arguments):
        raise ClusterConfigError("arguments must be a nonempty string list")
    if arguments[0] not in {"preflight", "pilot", "validate", "report"}:
        raise ClusterConfigError("only reviewed pipeline subcommands can be rendered")
    for item in arguments:
        _text(item, "arguments")
        if any(token in item for token in ("$(", "${", "`", ";", "|", "&")):
            raise ClusterConfigError("arguments contain shell-control syntax")
    output_values = []
    for index, item in enumerate(arguments[1:], start=1):
        if item == "--output":
            if index + 1 >= len(arguments):
                raise ClusterConfigError("command output value is missing")
            output_values.append(arguments[index + 1])
        elif item.startswith("--output="):
            output_values.append(item.split("=", 1)[1])
        elif item.startswith("--out"):
            raise ClusterConfigError("abbreviated or ambiguous output options are not accepted")
    if arguments[0] in {"preflight", "pilot", "report"}:
        if len(output_values) != 1:
            raise ClusterConfigError("command must declare exactly one output option")
        actual_output = _path(output_values[0], "command output")
        if actual_output == run or not _within(actual_output, run):
            raise ClusterConfigError("command output must be a new child of the configured personal run directory")
    elif output_values:
        raise ClusterConfigError("metadata validation does not accept an output option")
    command = [str(python), "-m", "ftd_prediction", *arguments]
    return "\n".join([
        "#!/bin/bash",
        "# Generated request only. Verify cluster configuration before submission.",
        f"# Memory units={config['memory_units']}; scope={config['memory_scope']}; user verified.",
        f"#BSUB -q {queue}", f"#BSUB -J {job}", f"#BSUB -n {cpus}",
        f"#BSUB -W {wall_time}", f'#BSUB -R "{resource}"',
        f"#BSUB -o {run}/%J.out", f"#BSUB -e {run}/%J.err", "",
        "set -euo pipefail", "umask 077", f"cd {shlex.quote(str(code))}",
        f"export PYTHONPATH={shlex.quote(str(code / 'src'))}",
        "exec " + shlex.join(command), "",
    ])


def render_lsf(config_path: str | Path, output_path: str | Path) -> Path:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    text = render_lsf_text(config)
    supplied = Path(output_path)
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError("job request already exists")
    destination = supplied.resolve()
    # Exclusive creation protects reviewed requests against accidental replacement.
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return destination
