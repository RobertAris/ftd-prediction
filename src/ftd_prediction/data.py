"""Read-only, bounded CSV interchange for small technical pilots."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ContractError, normalize_annotation, validate_manifest


MAX_DENSE_ELEMENTS = 100_000_000
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
CELL_COLUMNS = ("cell_id", "record_id", "cell_type", "layer", "x", "y", "tissue_component")
RECORD_CELL_COLUMNS = frozenset({
    "assay", "donor_id", "sample_id", "section_id", "retinal_region", "processing_batch",
    "source_label", "diagnosis_status", "mapping_status", "split",
})
INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class CellDataset:
    counts: np.ndarray
    genes: tuple[str, ...]
    cells: list[dict]
    manifest: dict
    input_paths: tuple[Path, ...]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("An input JSON object contains duplicate keys.")
        result[key] = value
    return result


def _read_manifest_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as error:
        raise ContractError("The study manifest could not be read.") from error
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ContractError("The manifest exceeds the supported metadata size.")
    return raw


def parse_manifest_bytes(raw: bytes) -> dict:
    """Validate one immutable byte snapshot, preserving duplicate-key checks."""
    if not isinstance(raw, bytes):
        raise ContractError("The manifest snapshot must be immutable bytes.")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ContractError("The manifest exceeds the supported metadata size.")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("The study manifest could not be read.") from error
    return validate_manifest(payload)


def read_manifest_snapshot(manifest_path) -> tuple[bytes, dict]:
    raw = _read_manifest_bytes(Path(manifest_path).resolve())
    return raw, parse_manifest_bytes(raw)


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"A {field} field is missing or malformed.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ContractError(f"A {field} field contains unsupported control characters.")
    return value


def _annotation(value: Any) -> str | None:
    return normalize_annotation(value)


def _coordinate(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError("A coordinate is malformed.") from error
    if not np.isfinite(number):
        raise ContractError("Coordinates must be finite.")
    return number


def _raw_count(value: str | None) -> int:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError("A raw-count entry is missing or malformed.")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ContractError("A raw-count entry is malformed.") from error
    if not number.is_finite() or number < 0 or number > INT64_MAX or number != number.to_integral_value():
        raise ContractError("Raw counts must be finite nonnegative integers within int64 range.")
    return int(number)


def _limits(max_cells: int, max_genes: int) -> None:
    if type(max_cells) is not int or type(max_genes) is not int or min(max_cells, max_genes) <= 0:
        raise ContractError("Cell and gene limits must be positive integers.")


def _read_cells(path: Path, *, max_cells: int) -> list[dict]:
    cells: list[dict] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            if header is None or len(header) != len(set(header)) or not set(CELL_COLUMNS).issubset(header):
                raise ContractError("The cell-metadata header is missing, incomplete or duplicated.")
            if set(header) - set(CELL_COLUMNS) - RECORD_CELL_COLUMNS:
                raise ContractError("The cell-metadata header contains unsupported columns.")
            for row in reader:
                if len(cells) >= max_cells:
                    raise ContractError("The cell input exceeds the configured cell limit.")
                if None in row or any(value is None for value in row.values()):
                    raise ContractError("A cell-metadata row has the wrong number of fields.")
                cell_id = _identifier(row["cell_id"], "cell_id")
                if cell_id in seen:
                    raise ContractError("Cell identifiers must be unique.")
                seen.add(cell_id)
                row["record_id"] = _identifier(row["record_id"], "record_id")
                row["cell_type"] = _annotation(row["cell_type"])
                row["layer"] = _annotation(row["layer"])
                row["tissue_component"] = _annotation(row["tissue_component"])
                row["x"], row["y"] = _coordinate(row["x"]), _coordinate(row["y"])
                if (row["x"] is None) != (row["y"] is None):
                    raise ContractError("A cell must supply both coordinates or neither.")
                cells.append(row)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ContractError("The cell-metadata input could not be read.") from error
    if not cells:
        raise ContractError("The cell-metadata input is empty.")
    return cells


def _read_counts(path: Path, cells: list[dict], *, max_genes: int) -> tuple[np.ndarray, tuple[str, ...]]:
    positions = {cell["cell_id"]: index for index, cell in enumerate(cells)}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or len(header) < 2 or header[0] != "cell_id":
                raise ContractError("The counts header must contain a cell identifier and genes.")
            genes = tuple(_identifier(gene, "gene_id") for gene in header[1:])
            if len(genes) != len(set(genes)) or "cell_id" in genes:
                raise ContractError("Gene identifiers must be unique and distinct from the cell identifier column.")
            if len(genes) > max_genes:
                raise ContractError("The count input exceeds the configured gene limit.")
            if len(cells) * len(genes) > MAX_DENSE_ELEMENTS:
                raise ContractError("The pilot matrix exceeds the dense allocation limit.")
            counts = np.empty((len(cells), len(genes)), dtype=np.int64)
            seen: set[str] = set()
            for row in reader:
                if len(row) != len(header):
                    raise ContractError("A count row has the wrong number of fields.")
                cell_id = _identifier(row[0], "cell_id")
                if cell_id in seen:
                    raise ContractError("The counts input contains duplicate cell identifiers.")
                if cell_id not in positions:
                    raise ContractError("Counts and metadata have different cell identifiers.")
                seen.add(cell_id)
                counts[positions[cell_id]] = [_raw_count(value) for value in row[1:]]
            if len(seen) != len(cells):
                raise ContractError("Counts and metadata have different cell identifiers.")
    except (OSError, UnicodeError, csv.Error) as error:
        raise ContractError("The count input could not be read.") from error
    counts.setflags(write=False)
    return counts, genes


def load_dataset(manifest_path, *, max_cells: int = 20000, max_genes: int = 5000,
                 manifest_bytes: bytes | None = None,
                 expected_input_paths: tuple[Path, ...] | None = None) -> CellDataset:
    """Read a bounded pilot dataset without changing any input file.

    This function does not establish permission to read study data. Execution
    context and study-data authorization are responsibilities of the runner.
    A runner may pass its already checked immutable manifest_bytes and resolved
    expected_input_paths. A changed live manifest or changed symlink resolution
    is rejected before opening any assay input.
    """
    _limits(max_cells, max_genes)
    path = Path(manifest_path).resolve()
    if manifest_bytes is None:
        manifest_bytes = _read_manifest_bytes(path)
    manifest = parse_manifest_bytes(manifest_bytes)
    if _read_manifest_bytes(path) != manifest_bytes:
        raise ContractError("The manifest changed after validation; start a new run.")
    counts_path = (path.parent / manifest["counts_file"]).resolve()
    cells_path = (path.parent / manifest["cells_file"]).resolve()
    if len({path, counts_path, cells_path}) != 3:
        raise ContractError("The manifest, counts and metadata must be distinct assets.")
    if expected_input_paths is not None and (path, counts_path, cells_path) != tuple(expected_input_paths):
        raise ContractError("Resolved input paths changed after validation; start a new run.")
    cells = _read_cells(cells_path, max_cells=max_cells)
    counts, genes = _read_counts(counts_path, cells, max_genes=max_genes)
    dataset = CellDataset(counts, genes, cells, manifest, (path, counts_path, cells_path))
    resolved_cells(dataset)
    return dataset


def resolved_cells(dataset: CellDataset) -> list[dict]:
    """Merge cell and record metadata while preserving verified boundaries."""
    manifest = validate_manifest(dataset.manifest)
    records = {record["record_id"]: record for record in manifest["records"]}
    result: list[dict] = []
    for cell in dataset.cells:
        record = records.get(cell.get("record_id"))
        if record is None:
            raise ContractError("A cell references an undeclared record.")
        for key in RECORD_CELL_COLUMNS:
            value = cell.get(key)
            if value not in (None, "") and value != record.get(key):
                raise ContractError("Cell metadata conflicts with its declared record.")
        cell_metadata = {key: value for key, value in cell.items() if key not in RECORD_CELL_COLUMNS}
        for field in ("cell_type", "layer", "tissue_component"):
            cell_metadata[field] = normalize_annotation(cell_metadata.get(field))
        merged = {**record, **cell_metadata}
        if record["assay"] == "xenium" and (cell.get("x") is None or cell.get("y") is None):
            raise ContractError("Xenium cells require finite coordinates.")
        merged["section_key"] = (record["donor_id"], record["sample_id"], record.get("section_id"))
        result.append(merged)
    if dataset.counts.ndim != 2 or dataset.counts.shape != (len(result), len(dataset.genes)):
        raise ContractError("The cell metadata and count matrix shapes disagree.")
    return result
