"""Offline technical reports drawn only from saved, explicitly scoped evidence."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .provenance import file_sha256, utc_now


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def inspect_run(path: str | Path) -> dict:
    supplied = Path(path).resolve()
    root = supplied.parent if supplied.name == "run.json" else supplied
    record = _read_json(root / "run.json")
    summary = _read_json(root / "summary.json")
    result = {
        "directory": str(root), "status": "missing_run_record", "data_kind": "unknown",
        "counts": {}, "integrity": [], "source_sha256": "unavailable", "seed": None,
        "started_at": None, "finished_at": None, "failure_type": None,
    }
    if record is None:
        result["integrity"].append("run.json is missing or unreadable")
        return result
    code = record.get("code") if isinstance(record.get("code"), dict) else {}
    result.update({
        "status": record.get("status", "unknown"), "data_kind": record.get("data_kind", "unknown"),
        "source_sha256": code.get("source_sha256", "unavailable"),
        "seed": record.get("seed"), "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"), "failure_type": record.get("failure_type"),
    })
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        result["integrity"].append("no completed artifact inventory")
        artifacts = []
    summary_verified = False
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            result["integrity"].append("invalid artifact inventory entry")
            continue
        relative = Path(artifact["path"])
        candidate = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in candidate.parents:
            result["integrity"].append("artifact path leaves run directory")
            continue
        if not candidate.is_file():
            result["integrity"].append("a declared artifact is missing")
            continue
        if file_sha256(candidate) != artifact.get("sha256"):
            result["integrity"].append("a declared artifact checksum differs")
            continue
        if relative.as_posix() == "summary.json":
            summary_verified = True
    if summary is None:
        result["integrity"].append("summary.json is missing or unreadable")
    elif not summary_verified:
        result["integrity"].append("summary is not verified by the saved artifact inventory")
    else:
        counts = summary.get("counts", {})
        if isinstance(counts, dict):
            result["counts"] = {
                key: value for key, value in counts.items()
                if key in {"n_cells", "n_genes", "n_donors", "n_sections"}
                and isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
    return result


def _flat(value: object) -> str:
    return " ".join(str(value).split())[:500]


def _markdown(value: object) -> str:
    return _flat(value).replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def render_markdown(runs: list[dict]) -> str:
    synthetic_only = bool(runs) and all(run["data_kind"] == "synthetic" for run in runs)
    title = "Synthetic pipeline technical report" if synthetic_only else "Retinal research technical report"
    lines = [f"# {title}", "", f"Generated: {utc_now()}", ""]
    if synthetic_only:
        lines += ["**All inputs in this report are generated synthetic fixtures. These checks contain no measured human biology or disease-performance evidence.**", ""]
    else:
        lines += ["**Private research report. Keep study-derived artifacts on approved storage until explicitly cleared for a destination.**", ""]
    lines += [
        "This report summarizes saved run status, input scope and verified artifact counts. Disease prediction, clinical calibration and replicated FTD effects are not established by these pilots.", "",
        "| Run | Status | Data kind | Cells | Genes | Donors | Physical sections |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for index, run in enumerate(runs, 1):
        values = [run["counts"].get(key, "unavailable") for key in ("n_cells", "n_genes", "n_donors", "n_sections")]
        lines.append("| " + " | ".join(_markdown(value) for value in [index, run["status"], run["data_kind"], *values]) + " |")
    lines += ["", "## Evidence and missing checks", ""]
    for index, run in enumerate(runs, 1):
        lines += [f"### Run {index}", "", f"- Saved run directory: {_markdown(run['directory'])}",
                  f"- Source digest: {_markdown(run['source_sha256'])}", f"- Seed: {_markdown(run['seed'])}",
                  f"- Started: {_markdown(run['started_at'])}", f"- Finished: {_markdown(run['finished_at'])}"]
        if run["failure_type"]:
            lines.append(f"- Failure type: {_markdown(run['failure_type'])}; inspect private logs before rerunning.")
        if run["integrity"]:
            lines.extend(f"- Evidence gap: {_markdown(issue)}." for issue in run["integrity"])
        else:
            lines.append("- Declared artifacts match saved checksums.")
        lines.append("")
    lines += [
        "## Interpretation and next steps", "",
        "Cells and sections are measurements nested within donors. Their counts are not the number of independent disease examples. Spatial summaries are section-local descriptors, and conditional-null residuals are not population disease scores.", "",
        "The current pilot does not establish annotation accuracy on human retina, optimal graph settings, a pretrained encoder comparison, tissue-mask safety without supplied masks, or a trained donor disease classifier. Record those as separate experiments with suitable inputs and acceptance evidence.", "",
        "Review failed/missing checks, inspect approved cluster-side figures and provenance, and decide the next bounded experiment before expanding compute.", "",
    ]
    return "\n".join(lines)


def render_html(runs: list[dict]) -> str:
    """Use escaped plain text and tables; no network assets or executable script."""
    synthetic_only = bool(runs) and all(run["data_kind"] == "synthetic" for run in runs)
    title = "Synthetic pipeline technical report" if synthetic_only else "Retinal research technical report"
    warning = "All inputs are synthetic fixtures; no human biology or disease-performance evidence." if synthetic_only else "Private research report; destination clearance is required before sharing."
    rows = []
    details = []
    for index, run in enumerate(runs, 1):
        values = [index, run["status"], run["data_kind"], *[run["counts"].get(key, "unavailable") for key in ("n_cells", "n_genes", "n_donors", "n_sections")]]
        rows.append("<tr>" + "".join(f"<td>{html.escape(_flat(value))}</td>" for value in values) + "</tr>")
        issues = run["integrity"] or ["Declared artifacts match saved checksums."]
        details.append(f"<section><h2>Run {index}</h2><p>Directory: {html.escape(run['directory'])}</p><p>Source digest: {html.escape(_flat(run['source_sha256']))}; seed: {html.escape(_flat(run['seed']))}</p><ul>" + "".join(f"<li>{html.escape(_flat(issue))}</li>" for issue in issues) + "</ul></section>")
    return "\n".join([
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#17212b}table{border-collapse:collapse;width:100%;display:block;overflow:auto}th,td{padding:.5rem;border:1px solid #b9c3cc;text-align:left}.scope{border-left:4px solid #af6a00;padding:.7rem;background:#fff4dc}section{overflow-wrap:anywhere}@media print{body{max-width:none;margin:0}table{display:table}}</style>",
        f"</head><body><h1>{html.escape(title)}</h1><p>Generated: {html.escape(utc_now())}</p>",
        f'<p class="scope"><strong>{html.escape(warning)}</strong></p>',
        "<p>Saved technical evidence does not establish a replicated FTD effect, donor disease prediction or clinical calibration.</p>",
        "<table><thead><tr><th>Run</th><th>Status</th><th>Data kind</th><th>Cells</th><th>Genes</th><th>Donors</th><th>Physical sections</th></tr></thead><tbody>",
        *rows, "</tbody></table>", *details,
        "<h2>Interpretation</h2><p>Cells and sections are nested within donors. Conditional spatial residuals describe a chosen null model. Check private logs and cluster-side outputs before selecting the next bounded experiment. These pilot commands do not complete the planned overnight single-cell and spatial model training package.</p>",
        "</body></html>", "",
    ])


def write_report(run_paths: list[str | Path], output_path: str | Path, *, output_format: str = "markdown") -> Path:
    if not run_paths:
        raise ValueError("at least one saved run is required")
    runs = [inspect_run(path) for path in run_paths]
    if output_format == "markdown":
        content = render_markdown(runs)
    elif output_format == "html":
        content = render_html(runs)
    else:
        raise ValueError("unsupported report format")
    supplied = Path(output_path)
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError("report already exists")
    destination = supplied.resolve()
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return destination
