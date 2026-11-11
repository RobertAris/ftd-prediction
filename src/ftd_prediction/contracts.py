"""Validated study metadata, with donor and physical-section boundaries explicit."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class ContractError(ValueError):
    """An input violates the declared, non-identifying analysis contract."""


ASSAYS = frozenset({"chromium_snrna", "chromium_scrna", "xenium"})
DATA_KINDS = frozenset({"synthetic", "study", "reference"})
SPLITS = frozenset({"train", "validation", "test", "reference"})
MAPPING_STATUSES = frozenset({"confirmed", "unresolved", "conflicting"})
DIAGNOSIS_STATUSES = frozenset({"confirmed", "provisional", "unknown"})
MISSING_TOKENS = frozenset({"unknown", "unresolved", "not_applicable", "restricted"})
BASE_RECORD_FIELDS = (
    "record_id", "assay", "donor_id", "sample_id", "retinal_region",
    "processing_batch", "source_label", "diagnosis_status", "label_evidence_ref",
    "mapping_status", "mapping_evidence_ref",
)
SPATIAL_FIELDS = (
    "section_id", "acquisition_region_id", "panel_ref", "segmentation_ref",
    "coordinate_units", "coordinate_transform_ref",
)


def _text(value: Any, *, field: str, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"A required {field} field is missing or malformed.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ContractError(f"A {field} field contains unsupported control characters.")
    return value


def is_resolved(value: Any) -> bool:
    """Whether a metadata value is explicit rather than a missing-value token."""
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in MISSING_TOKENS


def normalize_annotation(value: Any) -> str | None:
    """Represent every declared annotation missing token as a missing value."""
    if value is None:
        return None
    if isinstance(value, str) and (not value.strip() or value.strip().lower() in MISSING_TOKENS):
        return None
    return _text(value, field="annotation")


def validate_donor_splits(records: list[dict]) -> None:
    """Reject inconsistent split assignment across every record for a donor."""
    if not isinstance(records, list):
        raise ContractError("Manifest records must be a list.")
    assignments: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("Every manifest record must be an object.")
        donor = _text(record.get("donor_id"), field="donor_id")
        split = record.get("split")
        if split is None:
            continue
        if not isinstance(split, str) or split not in SPLITS:
            raise ContractError("A donor split is unsupported.")
        if not is_resolved(donor):
            raise ContractError("An unresolved donor cannot be assigned to a split.")
        if donor in assignments and assignments[donor] != split:
            raise ContractError("A donor occurs in more than one split.")
        assignments[donor] = split


def validate_manifest(payload: dict) -> dict:
    """Copy and validate schema v1 without reading any assay files.

    Missing metadata tokens remain explicit. Such records can be inventoried,
    but graph-ready checks and quantitative summaries impose stricter criteria.
    Source labels are literal: no diagnosis harmonization is performed.
    """
    if not isinstance(payload, dict):
        raise ContractError("The study manifest must be an object.")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ContractError("Unsupported manifest schema version.")
    if not isinstance(payload.get("data_kind"), str) or payload["data_kind"] not in DATA_KINDS:
        raise ContractError("The manifest data kind is unsupported.")
    for field in ("dataset_id", "source_version", "counts_file", "cells_file"):
        _text(payload.get(field), field=field)
    if payload["counts_file"] == payload["cells_file"]:
        raise ContractError("Counts and cell metadata must be distinct input assets.")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("The manifest needs at least one record.")
    normalized = deepcopy(payload)
    seen_ids: set[str] = set()
    physical_units: dict[tuple[str, str, str | None], str] = {}
    labels: dict[tuple[str, str], str] = {}
    for record in normalized["records"]:
        if not isinstance(record, dict):
            raise ContractError("Every manifest record must be an object.")
        for field in BASE_RECORD_FIELDS:
            if field == "retinal_region":
                continue
            _text(record.get(field), field=field)
        # Unknown comparability remains explicit and is handled at record level
        # by summaries. No anatomical region is inferred from another field.
        region = record.get("retinal_region")
        if region is None or (isinstance(region, str) and not region.strip()):
            record["retinal_region"] = None
        else:
            _text(region, field="retinal_region")
        if record["record_id"] in seen_ids:
            raise ContractError("Manifest record identifiers must be unique.")
        seen_ids.add(record["record_id"])
        if record["assay"] not in ASSAYS:
            raise ContractError("An assay is unsupported.")
        if record["mapping_status"] not in MAPPING_STATUSES:
            raise ContractError("A mapping status is unsupported.")
        if record["diagnosis_status"] not in DIAGNOSIS_STATUSES:
            raise ContractError("A diagnosis status is unsupported.")
        record.setdefault("split", None)
        if record["assay"] == "xenium":
            for field in SPATIAL_FIELDS:
                value = record.get(field)
                if value is not None:
                    _text(value, field=field)
                record.setdefault(field, None)
        else:
            # Spatial provenance on RNA is not interpreted as a Xenium section.
            record.setdefault("section_id", None)
            if record["section_id"] not in (None, "not_applicable"):
                raise ContractError("RNA records cannot declare spatial section identity.")
            record["section_id"] = None
        if is_resolved(record["donor_id"]):
            label_key = (record["assay"], record["donor_id"])
            if label_key in labels and labels[label_key] != record["source_label"]:
                raise ContractError("Source labels conflict within an assay and donor.")
            labels[label_key] = record["source_label"]
        section = record.get("section_id")
        # A sample/section is represented by one canonical input record. This
        # prevents technical reruns from silently becoming extra observations.
        if record["assay"] != "xenium" or is_resolved(section):
            unit_key = (record["assay"], record["sample_id"], section)
            if unit_key in physical_units:
                if physical_units[unit_key] != record["donor_id"]:
                    raise ContractError("A physical sampling unit has contradictory donor mappings.")
                raise ContractError("A physical sampling unit has duplicate canonical records.")
            physical_units[unit_key] = record["donor_id"]
    validate_donor_splits(normalized["records"])
    return normalized


def require_graph_ready(manifest: dict) -> None:
    """Require explicit, confirmed physical mappings before spatial graph work."""
    validated = validate_manifest(manifest)
    for record in validated["records"]:
        if record["assay"] != "xenium":
            raise ContractError("Graph construction requires a Xenium-only manifest.")
        if record["mapping_status"] != "confirmed":
            raise ContractError("Graph construction requires confirmed physical mappings.")
        for field in ("donor_id", "sample_id", "mapping_evidence_ref", *SPATIAL_FIELDS):
            if not is_resolved(record.get(field)):
                raise ContractError("Graph construction requires complete spatial provenance.")
        if record["coordinate_units"] != "um":
            raise ContractError("Graph coordinates must be explicitly transformed to micrometres.")
