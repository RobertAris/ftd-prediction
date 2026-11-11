"""Render a reviewed LSF request without connecting to or submitting a job."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path


class ClusterConfigError(ValueError):
    pass


_PLACEHOLDER = re.compile(r"replace|unverified|placeholder|<|>|your[_ -]|example", re.IGNORECASE)
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


def _command(config: dict, run: Path, python: Path) -> list[str]:
    module = config.get("module", "ftd_prediction")
    schemas = {
        "ftd_prediction": {
            "preflight": ({"--output", "--config", "--seed"}, {"--output"}),
            "pilot": ({"--output", "--config", "--seed", "--manifest", "--mode",
                       "--execution-context", "--canonical-input-root"}, {"--output", "--manifest"}),
            "validate": ({"--manifest"}, {"--manifest"}),
            "report": ({"--run", "--output", "--format"}, {"--run", "--output"}),
        },
        "ftd_prediction.descriptive": None,
        "ftd_prediction.neural": None,
        "ftd_prediction.neural_report": None,
        "ftd_prediction.descriptive_report": None,
        "ftd_prediction.molecular": None,
    }
    if not isinstance(module, str) or module not in schemas:
        raise ClusterConfigError("unsupported module")
    arguments = config.get("arguments")
    if not isinstance(arguments, list) or not arguments or not all(isinstance(v, str) for v in arguments):
        raise ClusterConfigError("arguments must be a nonempty string list")
    for item in arguments:
        _text(item, "arguments")
        if any(token in item for token in ("$(", "${", "`", ";", "|", "&")):
            raise ClusterConfigError("arguments contain shell-control syntax")
    if module == "ftd_prediction":
        if arguments[0] not in schemas[module]:
            raise ClusterConfigError("only reviewed pipeline subcommands can be rendered")
        allowed, required = schemas[module][arguments[0]]
        options = arguments[1:]
    elif module in {"ftd_prediction.neural_report", "ftd_prediction.descriptive_report"}:
        allowed = required = {"--run", "--output"}
        options = arguments
    else:
        allowed = {"--config", "--output", "--mode", "--execution-context"}
        required = {"--config", "--output"}
        options = arguments
    values = {}
    index = 0
    while index < len(options):
        flag, equal, value = options[index].partition("=")
        if flag not in allowed:
            raise ClusterConfigError("unknown or abbreviated option")
        if not equal:
            index += 1
            if index == len(options):
                raise ClusterConfigError("option value is missing")
            value = options[index]
        if not value or value.startswith("-"):
            raise ClusterConfigError("option value is missing or unsafe")
        repeatable = {"--canonical-input-root"}
        if module in {"ftd_prediction", "ftd_prediction.neural_report"}:
            repeatable.add("--run")
        if flag in values and flag not in repeatable:
            raise ClusterConfigError("duplicate option")
        values.setdefault(flag, []).append(value)
        if flag in {"--output", "--config", "--manifest", "--run", "--canonical-input-root"}:
            if flag == "--output" or value.startswith("/") or module != "ftd_prediction":
                _path(value, flag)
            elif not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or ".." in Path(value).parts:
                raise ClusterConfigError("input option must be a plain path")
        elif flag == "--seed" and not re.fullmatch(r"[0-9]+", value):
            raise ClusterConfigError("seed must be a nonnegative integer")
        elif flag == "--mode":
            modes = ({"study", "synthetic"} if module in {
                "ftd_prediction.descriptive", "ftd_prediction.molecular"}
                else {"study", "reference", "synthetic"})
            if value not in modes:
                raise ClusterConfigError("unsupported execution mode")
        elif flag == "--execution-context" and value not in {"pmacs", "local"}:
            raise ClusterConfigError("unsupported execution context")
        elif flag == "--format" and value not in {"markdown", "html"}:
            raise ClusterConfigError("unsupported report format")
        index += 1
    if required - values.keys():
        raise ClusterConfigError("required command options are missing")
    if "--output" in values:
        output = _path(values["--output"][0], "command output")
        if output == run or not _within(output, run):
            raise ClusterConfigError("command output must be a new child of the configured personal run directory")
    return [str(python), "-m", module, *arguments]


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
    command = _command(config, run, python)
    return "\n".join([
        "#!/bin/bash",
        "# Generated request only. Verify cluster configuration before submission.",
        f"# Memory units={config['memory_units']}; scope={config['memory_scope']}; user verified.",
        f"#BSUB -q {queue}", f"#BSUB -J {job}", f"#BSUB -n {cpus}",
        f"#BSUB -W {wall_time}", f'#BSUB -R "{resource}"',
        f"#BSUB -o {run}/%J.out", f"#BSUB -e {run}/%J.err", "",
        "set -euo pipefail", "umask 077", f"cd {shlex.quote(str(code))}",
        f"export PYTHONPATH={shlex.quote(str(code / 'src'))}",
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1",
        f"export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} NUMEXPR_NUM_THREADS={cpus}",
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
