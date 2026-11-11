"""Command line runner for validated, bounded retinal research pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

from .provenance import environment_record, file_sha256, input_record, source_snapshot, utc_now, write_json


DEFAULT_CONFIG = {
    "seed": 11, "max_cells": 20000, "max_genes": 5000,
    "radius_um": 22.0, "max_neighbors": 8, "n_permutations": 32,
    "canonical_input_roots": [],
}

_SAFE_RUNTIME_MESSAGES = {
    "pilot configuration contains unsupported fields",
    "pilot integer configuration is invalid",
    "radius_um must be finite and positive",
    "canonical_input_roots must be a path list",
    "input data kind requires a matching explicit execution mode",
    "non-synthetic processing requires the declared PMACS execution context",
    "non-synthetic processing requires an LSF compute-job context",
    "output directory overlaps a canonical input root",
    "run directory already exists; choose a unique run ID",
    "non-synthetic processing requires explicit canonical input roots",
    "assay inputs must resolve inside the declared canonical input roots",
    "output directory overlaps an input file",
    "partial tissue-component metadata must be resolved before graph analysis",
    "input fingerprint changed during analysis",
    "manifest changed after validation; start a new run with immutable inputs",
    "loaded input contract differs from the validated manifest snapshot",
}


class CommandLineError(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default error includes the submitted value. Keep argument
        # mistakes useful without echoing identifiers or private source paths.
        raise CommandLineError("Invalid or missing command arguments; use the command's --help for accepted options.")


def load_config(path: str | Path | None, *, seed: int | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            supplied = json.load(handle)
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
            raise ValueError("pilot configuration contains unsupported fields")
        config.update(supplied)
    if seed is not None:
        config["seed"] = seed
    for key in ("seed", "max_cells", "max_genes", "max_neighbors", "n_permutations"):
        value = config[key]
        minimum = 0 if key == "seed" else 2 if key == "n_permutations" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("pilot integer configuration is invalid")
    radius = config["radius_um"]
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius_um must be finite and positive")
    roots = config["canonical_input_roots"]
    if not isinstance(roots, list) or not all(isinstance(item, str) and item for item in roots):
        raise ValueError("canonical_input_roots must be a path list")
    return config


def validate_runtime_scope(manifest: dict, *, mode: str, execution_context: str, environ: dict | None = None) -> None:
    """Safe defaults only: these checks do not grant institutional permission."""
    kind = manifest.get("data_kind")
    if kind != mode:
        raise ValueError("input data kind requires a matching explicit execution mode")
    if kind != "synthetic":
        if execution_context != "pmacs":
            raise ValueError("non-synthetic processing requires the declared PMACS execution context")
        job = (os.environ if environ is None else environ).get("LSB_JOBID", "")
        if not isinstance(job, str) or not job.isdigit() or int(job) <= 0:
            raise ValueError("non-synthetic processing requires an LSF compute-job context")


def protect_output_path(output: str | Path, canonical_roots: list[str | Path]) -> Path:
    supplied = Path(output)
    destination = supplied.resolve()
    for value in canonical_roots:
        root = Path(value).resolve()
        if destination == root or root in destination.parents:
            raise ValueError("output directory overlaps a canonical input root")
    if supplied.exists() or supplied.is_symlink() or destination.exists():
        raise FileExistsError("run directory already exists; choose a unique run ID")
    return destination


def _logger(directory: Path) -> logging.Logger:
    logger = logging.getLogger(f"ftd_prediction.run.{directory}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(directory / "run.log", mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def run_pilot(
    manifest_path: str | Path, output: str | Path, *, config: dict | None = None,
    mode: str = "synthetic", execution_context: str = "local",
    command: list[str] | None = None,
) -> Path:
    """Write private artifacts in a new run directory and retain failure records."""
    from .contracts import is_resolved, require_graph_ready, validate_manifest
    from .data import load_dataset, read_manifest_snapshot, resolved_cells
    from .rna import summarize_expression

    settings = dict(DEFAULT_CONFIG if config is None else config)
    manifest_file = Path(manifest_path).resolve()
    manifest_bytes, manifest = read_manifest_snapshot(manifest_file)
    validated_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    validate_runtime_scope(manifest, mode=mode, execution_context=execution_context)
    canonical_roots = settings.get("canonical_input_roots", [])
    if mode != "synthetic" and not canonical_roots:
        raise ValueError("non-synthetic processing requires explicit canonical input roots")
    destination = protect_output_path(output, canonical_roots)
    # Independently protect the actual input files even if no root was configured.
    protected_files = [manifest_file]
    for field in ("counts_file", "cells_file"):
        path = Path(manifest[field])
        protected_files.append((manifest_file.parent / path).resolve() if not path.is_absolute() else path.resolve())
    if mode != "synthetic":
        roots = [Path(value).resolve() for value in canonical_roots]
        if any(not any(path == root or root in path.parents for root in roots) for path in protected_files[1:]):
            raise ValueError("assay inputs must resolve inside the declared canonical input roots")
    if any(destination == path or path in destination.parents for path in protected_files):
        raise ValueError("output directory overlaps an input file")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    logger = _logger(destination)
    record = {
        "schema_version": 1, "status": "running", "started_at": utc_now(),
        "data_kind": mode, "execution_context": execution_context,
        "lsf_job_id": os.environ.get("LSB_JOBID") if execution_context == "pmacs" else None,
        "seed": settings["seed"], "configuration": settings,
        "validated_manifest_sha256": validated_manifest_sha256,
        "command": command if command is not None else [],
        "code": source_snapshot(), "environment": environment_record(),
        "output_directory": str(destination), "disease_inference_enabled": False,
        "evidence_scope": "synthetic_pipeline_check" if mode == "synthetic" else "descriptive_specimen_analysis",
    }
    write_json(destination / "run.json", record)
    try:
        logger.info("Starting validated bounded pilot; data_kind=%s", mode)
        before = input_record(protected_files)
        if file_sha256(manifest_file) != validated_manifest_sha256:
            raise RuntimeError("manifest changed after validation; start a new run with immutable inputs")
        record["inputs"] = before
        write_json(destination / "run.json", record, exclusive=False)
        dataset = load_dataset(manifest_file, max_cells=settings["max_cells"], max_genes=settings["max_genes"],
                               manifest_bytes=manifest_bytes, expected_input_paths=tuple(protected_files))
        if dataset.manifest != manifest or set(Path(path).resolve() for path in dataset.input_paths) != set(protected_files):
            raise RuntimeError("loaded input contract differs from the validated manifest snapshot")
        if file_sha256(manifest_file) != validated_manifest_sha256:
            raise RuntimeError("manifest changed after validation; start a new run with immutable inputs")
        cells = resolved_cells(dataset)
        expression = summarize_expression(dataset)
        expression_path = destination / "expression.json"
        write_json(expression_path, expression)
        artifacts = [{"name": "expression", "path": expression_path.name, "sha256": input_record([expression_path])[0]["sha256"]}]
        assays = {cell["assay"] for cell in cells}
        spatial_summary = {"status": "not_applicable"}
        if "xenium" in assays:
            from .spatial import build_radius_graph, conditional_neighborhood_residuals, neighborhood_fractions, pool_sections_to_donors

            require_graph_ready(dataset.manifest)
            coordinates = np.asarray([[cell["x"], cell["y"]] for cell in cells], dtype=float)
            section_keys = [(cell["donor_id"], cell["sample_id"], cell["section_id"]) for cell in cells]
            labels = [cell["cell_type"] for cell in cells]
            components = [cell.get("tissue_component") for cell in cells]
            if any(value is not None for value in components) and not all(value is not None for value in components):
                raise ValueError("partial tissue-component metadata must be resolved before graph analysis")
            # Missing component labels must not be mistaken for a verified barrier.
            component_argument = components if all(value is not None for value in components) else None
            edges = build_radius_graph(
                coordinates, section_keys, radius_um=settings["radius_um"],
                max_neighbors=settings["max_neighbors"], cell_ids=[cell["cell_id"] for cell in cells],
                tissue_components=component_argument,
            )
            labels_available = any(value is not None for value in labels)
            if labels_available:
                neighborhoods = neighborhood_fractions(edges, labels)
                has_neighbors = ~np.asarray(neighborhoods["is_isolate"], dtype=bool)
            else:
                degree = np.bincount(edges[0], minlength=len(cells))
                has_neighbors = degree > 0
                neighborhoods = {
                    "status": "unavailable", "reason": "cell_type_annotations_missing",
                    "degree": degree, "is_isolate": ~has_neighbors,
                    "n_unknown_cells": len(cells),
                }
            spatial_summary = {
                "status": "complete", "edge_count": int(edges.shape[1]),
                "directed_graph": True, "radius_um": settings["radius_um"],
                "max_neighbors": settings["max_neighbors"],
                "mask_validation": "not_supplied", "component_validation": component_argument is not None,
                "neighborhoods": neighborhoods,
                "cell_type_features_status": "available" if labels_available else "unavailable",
                "annotated_cell_fraction": sum(value is not None for value in labels) / len(labels),
                "n_isolates": int((~has_neighbors).sum()),
                "isolate_fraction": float((~has_neighbors).mean()),
                "neighborhood_means_condition_on_has_neighbors": True,
                "interpretation": "section_local_spatial_descriptors",
            }
            if labels_available and all(cell.get("layer") is not None for cell in cells):
                strata = [(*key, cell["layer"]) for key, cell in zip(section_keys, cells)]
                spatial_summary["conditional_null"] = conditional_neighborhood_residuals(
                    edges, labels, strata, n_permutations=settings["n_permutations"],
                    seed=settings["seed"], categories=neighborhoods["categories"],
                )
                null = spatial_summary["conditional_null"]
                null["status"] = "available"
                zero_variance = np.asarray(null["zero_variance"], dtype=bool)
                null["standardized_residual_available"] = (~zero_variance) & has_neighbors[:, None]
                null["degenerate_null_fraction"] = float(zero_variance.mean()) if zero_variance.size else None
                null["standardized_residual_available_fraction"] = float(null["standardized_residual_available"].mean()) if zero_variance.size else None
            else:
                spatial_summary["conditional_null"] = {
                    "status": "unavailable", "reason": "layer_annotations_missing" if labels_available else "cell_type_annotations_missing",
                    "interpretation": "no_anatomy_conditioned_inference",
                }
            # Pool only observed neighborhoods. Zero placeholders for isolates
            # must not become neutral biological values. Qualify section tokens
            # by sample to avoid merging distinct physical sections.
            region_available = np.asarray([is_resolved(cell.get("retinal_region")) for cell in cells], dtype=bool)
            eligible = np.flatnonzero(has_neighbors & region_available) if labels_available else np.array([], dtype=int)
            pooled = pool_sections_to_donors(
                np.asarray(neighborhoods["fractions"])[eligible],
                [cells[index]["donor_id"] for index in eligible],
                [json.dumps([cells[index]["sample_id"], cells[index]["section_id"]], separators=(",", ":")) for index in eligible],
                [cells[index]["retinal_region"] for index in eligible],
            ) if len(eligible) else []
            by_group = {(row["donor_id"], row["retinal_region"]): row for row in pooled}
            groups = {}
            for index, cell in enumerate(cells):
                key = ("donor_region", cell["donor_id"], cell["retinal_region"], None) if region_available[index] else ("record", cell["donor_id"], cell.get("retinal_region"), cell["record_id"])
                groups.setdefault(key, []).append(index)
            for group in sorted(groups, key=repr):
                indices = groups[group]
                present = int(has_neighbors[indices].sum())
                resolved_region = group[0] == "donor_region"
                row = by_group.get((group[1], group[2])) if resolved_region else None
                if row is None:
                    row = {"donor_id": group[1], "retinal_region": group[2], "n_sections": 0,
                           "n_cells": 0, "section_ids": [], "mean": None}
                    pooled.append(row)
                unavailable_reasons = []
                if not labels_available:
                    unavailable_reasons.append("cell_type_annotations_missing")
                if not resolved_region:
                    unavailable_reasons.append("retinal_region_unresolved")
                if not present:
                    unavailable_reasons.append("no_neighbors")
                status = "available"
                if not labels_available:
                    status = "unavailable_cell_types"
                elif not resolved_region:
                    status = "unavailable_region_comparability"
                elif not present:
                    status = "unavailable_no_neighbors"
                row.update({"status": status, "unavailable_reasons": unavailable_reasons,
                            "pooling_scope": group[0], "record_ids": sorted({cells[index]["record_id"] for index in indices}),
                            "n_total_cells": len(indices), "n_cells_with_neighbors": present,
                            "n_isolates": len(indices) - present,
                            "isolate_fraction": (len(indices) - present) / len(indices),
                            "n_total_sections": len({section_keys[index] for index in indices}),
                            "mean_conditions_on_has_neighbors": True})
            spatial_summary["pooled_neighborhoods"] = pooled
            spatial_summary["pooling_available_groups"] = sum(row["status"] == "available" for row in pooled)
            spatial_path = destination / "spatial.json"
            write_json(spatial_path, spatial_summary)
            artifacts.append({"name": "spatial", "path": spatial_path.name, "sha256": input_record([spatial_path])[0]["sha256"]})
        after = input_record(protected_files)
        if before != after:
            raise RuntimeError("input fingerprint changed during analysis")
        summary = {
            "schema_version": 1, "status": "complete", "data_kind": mode,
            "evidence_scope": record["evidence_scope"], "disease_inference_enabled": False,
            "counts": {
                "n_cells": len(cells), "n_genes": len(dataset.genes),
                "n_donors": len({cell["donor_id"] for cell in cells}),
                "n_sections": len({(cell["donor_id"], cell["sample_id"], cell.get("section_id")) for cell in cells if cell["assay"] == "xenium"}),
            },
            "spatial": {key: spatial_summary[key] for key in ("status", "edge_count", "mask_validation", "cell_type_features_status", "annotated_cell_fraction", "n_isolates", "isolate_fraction", "pooling_available_groups") if key in spatial_summary},
            "artifacts": artifacts,
            "limitations": [
                "No disease prediction performance is established.",
                "No tissue-mask barrier validation is available unless supplied by a later reviewed adapter.",
                "Cell/section variation is not independent disease-donor replication.",
            ],
        }
        write_json(destination / "summary.json", summary)
        record.update({"status": "complete", "finished_at": utc_now(), "inputs_unchanged": True,
                       "artifacts": artifacts + [{"name": "summary", "path": "summary.json", "sha256": input_record([destination / "summary.json"])[0]["sha256"]}]})
        write_json(destination / "run.json", record, exclusive=False)
        logger.info("Pilot completed; artifact metadata saved")
    except Exception as error:
        record.update({"status": "failed", "finished_at": utc_now(), "failure_type": type(error).__name__,
                       "failure_message": "See private run.log for diagnostic details."})
        write_json(destination / "run.json", record, exclusive=False)
        logger.exception("Pilot failed; private diagnostics follow")
        raise
    finally:
        _close_logger(logger)
    return destination


def run_preflight(output: str | Path, *, config: dict | None = None, command: list[str] | None = None) -> Path:
    from .synthetic import generate_synthetic_fixtures

    settings = dict(DEFAULT_CONFIG if config is None else config)
    destination = protect_output_path(output, settings.get("canonical_input_roots", []))
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    status = {"schema_version": 1, "status": "running", "data_kind": "synthetic",
              "evidence_scope": "synthetic_pipeline_check", "started_at": utc_now(), "seed": settings["seed"]}
    write_json(destination / "preflight.json", status)
    try:
        manifests = generate_synthetic_fixtures(destination / "fixtures", seed=settings["seed"])
        runs = {}
        for name, manifest in manifests.items():
            context = "pmacs" if os.environ.get("LSB_JOBID", "").isdigit() else "local"
            runs[name] = str(run_pilot(manifest, destination / name, config=settings, command=command, execution_context=context))
        status.update({"status": "complete", "finished_at": utc_now(), "runs": runs,
                       "disease_performance_claim": False})
        write_json(destination / "preflight.json", status, exclusive=False)
        from .reporting import write_report
        write_report(list(runs.values()), destination / "report.md", output_format="markdown")
        write_report(list(runs.values()), destination / "report.html", output_format="html")
    except Exception as error:
        status.update({"status": "failed", "finished_at": utc_now(), "failure_type": type(error).__name__})
        write_json(destination / "preflight.json", status, exclusive=False)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="ftd-prediction", allow_abbrev=False, description="Donor-aware retinal research pilots; synthetic evidence is labelled explicitly.")
    commands = parser.add_subparsers(dest="subcommand", required=True)
    validate = commands.add_parser("validate", allow_abbrev=False, help="Validate manifest metadata without loading assay data")
    validate.add_argument("--manifest", type=Path, required=True)
    for name in ("preflight", "pilot"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--config", type=Path)
        command.add_argument("--seed", type=int)
        if name == "pilot":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--mode", choices=("synthetic", "study", "reference"), default="synthetic")
            command.add_argument("--execution-context", choices=("local", "pmacs"), default="local")
            command.add_argument("--canonical-input-root", action="append", default=[])
    render = commands.add_parser("render-lsf", allow_abbrev=False, help="Render only; never connect or submit")
    render.add_argument("--config", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", allow_abbrev=False, help="Render an offline technical report from saved run evidence")
    report.add_argument("--run", type=Path, action="append", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--format", choices=("markdown", "html"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
        if args.subcommand == "validate":
            from .data import read_manifest_snapshot
            _, manifest = read_manifest_snapshot(args.manifest)
            print(json.dumps({"status": "manifest_valid", "data_kind": manifest["data_kind"], "assay_files_loaded": False}))
        elif args.subcommand in {"preflight", "pilot"}:
            config = load_config(args.config, seed=args.seed)
            if args.subcommand == "preflight":
                output = run_preflight(args.output, config=config, command=["ftd-prediction", *arguments])
            else:
                config["canonical_input_roots"] = [*config["canonical_input_roots"], *args.canonical_input_root]
                output = run_pilot(args.manifest, args.output, config=config, mode=args.mode,
                                   execution_context=args.execution_context, command=["ftd-prediction", *arguments])
            print(json.dumps({"status": "complete", "output_directory": str(output)}))
        elif args.subcommand == "render-lsf":
            from .cluster import render_lsf
            output = render_lsf(args.config, args.output)
            print(json.dumps({"status": "rendered_only", "job_submitted": False, "output": str(output)}))
        elif args.subcommand == "report":
            from .reporting import write_report
            output = write_report(args.run, args.output, output_format=args.format)
            print(json.dumps({"status": "report_written", "output": str(output)}))
    except Exception as error:
        # Detailed exceptions can contain private source paths or rows. Pilot
        # diagnostics are retained only in the private run directory.
        reason = "processing_failed"
        message = "Command failed. Inspect any private run.log; input values are withheld."
        if isinstance(error, CommandLineError):
            reason, message = "invalid_command_arguments", str(error)
        elif type(error).__name__ == "ContractError" and type(error).__module__ == "ftd_prediction.contracts":
            reason, message = "manifest_contract_invalid", str(error)
        elif type(error).__name__ == "ClusterConfigError" and type(error).__module__ == "ftd_prediction.cluster":
            reason, message = "cluster_config_invalid", str(error)
        elif isinstance(error, json.JSONDecodeError):
            reason, message = "invalid_json", "Configuration or manifest JSON is invalid; check its syntax."
        elif isinstance(error, FileNotFoundError):
            reason, message = "required_path_missing", "A required file or directory is missing; verify the configured paths."
        elif isinstance(error, FileExistsError):
            reason, message = "output_exists", "The output already exists; choose a new run or report path."
        elif isinstance(error, PermissionError):
            reason, message = "permission_denied", "Read/write permission is missing; check the approved input and personal workspace access."
        elif str(error) in _SAFE_RUNTIME_MESSAGES:
            reason, message = "execution_configuration_invalid", str(error)
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "reason_code": reason, "message": message}), file=sys.stderr)
        return 1
    return 0
