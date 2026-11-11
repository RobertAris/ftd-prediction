"""Descriptive expression summaries with donor, region and assay retained."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from .contracts import ContractError, is_resolved
from .data import CellDataset, resolved_cells


NORMALIZATION_TOTAL = 10_000.0


def summarize_expression(dataset: CellDataset) -> dict:
    """Sum raw counts by assay/donor/region/type, without disease inference.

    Pseudobulk counts are summed with Python integers to avoid int64 wraparound.
    log1p expression uses total-count normalization of each aggregate to 10,000.
    Missing annotations have an explicit row and flag, not an invented type.
    Missing populations elsewhere are not emitted as measured all-zero profiles.
    Unknown/missing retinal regions retain separate record-level rows because
    their comparability has not been established.
    """
    cells = resolved_cells(dataset)
    counts = np.asarray(dataset.counts)
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ContractError("Expression summaries require nonnegative integer raw counts.")
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        if cell["mapping_status"] != "confirmed" or not is_resolved(cell["donor_id"]):
            raise ContractError("Donor summaries require confirmed donor mappings.")
        region = cell.get("retinal_region")
        record_scope = None if is_resolved(region) else cell["record_id"]
        groups[(cell["assay"], cell["donor_id"], region, cell.get("cell_type"), record_scope)].append(index)
    rows = []
    order = lambda key: tuple("" if value is None else str(value) for value in key)
    for (assay, donor, region, cell_type, record_scope), indices in sorted(groups.items(), key=lambda item: order(item[0])):
        raw = [sum(int(counts[index, gene]) for index in indices) for gene in range(counts.shape[1])]
        library_total = sum(raw)
        normalized = [math.log1p(value / library_total * NORMALIZATION_TOTAL) for value in raw] if library_total else [None] * len(raw)
        rows.append({
            "assay": assay,
            "donor_id": donor,
            "retinal_region": region,
            "region_resolved": is_resolved(region),
            "pooling_scope": "donor_region" if record_scope is None else "record",
            "record_ids": sorted({cells[index]["record_id"] for index in indices}),
            "cell_type": cell_type,
            "cell_type_missing": cell_type is None,
            "n_cells": len(indices),
            "library_total": library_total,
            "zero_library": library_total == 0,
            "raw_counts": raw,
            "log1p_normalized": normalized,
            "source_labels": sorted({cells[index]["source_label"] for index in indices}),
        })
    return {
        "genes": list(dataset.genes),
        "normalization_total": NORMALIZATION_TOTAL,
        "rows": rows,
        "evidence_scope": "descriptive_specimen_summary",
    }
