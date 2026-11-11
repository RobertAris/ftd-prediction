"""Generate deliberately artificial, unpaired cohorts for executable checks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .provenance import write_json


GENES = ("SYN_ROD", "SYN_GLIA", "SYN_STATE", "SYN_CONST", "SYN_A", "SYN_B")


def _record(assay: str, donor: str, sample: str, record_id: str, section: str | None) -> dict:
    result = {
        "record_id": record_id, "assay": assay, "donor_id": donor,
        "sample_id": sample, "retinal_region": "synthetic_region",
        "processing_batch": "synthetic_batch", "source_label": "synthetic_unlabelled",
        "diagnosis_status": "unknown", "label_evidence_ref": "synthetic_generator_v1",
        "mapping_status": "confirmed", "mapping_evidence_ref": "synthetic_generator_v1",
        "split": None,
    }
    if assay == "xenium":
        result.update({
            "section_id": section, "acquisition_region_id": f"SYN_ACQ_{donor}",
            "panel_ref": "synthetic_panel_v1", "segmentation_ref": "synthetic_grid_v1",
            "coordinate_units": "um", "coordinate_transform_ref": "synthetic_identity_v1",
        })
    return result


def _write_cohort(path: Path, assay: str, rng: np.random.Generator) -> Path:
    path.mkdir()
    records, cells, counts = [], [], []
    donor_count = 2 if assay != "xenium" else 3
    sections = 1 if assay != "xenium" else 2
    prefix = "SYN_RNA" if assay != "xenium" else "SYN_SPATIAL"
    for donor_index in range(donor_count):
        donor = f"{prefix}_D{donor_index + 1}"
        for section_index in range(sections):
            sample = f"{donor}_S{section_index + 1}"
            section = f"SYN_SECTION_{section_index + 1}" if assay == "xenium" else None
            record_id = f"{sample}_R1"
            records.append(_record(assay, donor, sample, record_id, section))
            for cell_index in range(24):
                cell_id = f"{record_id}_C{cell_index + 1}"
                cell_type = "Rod" if cell_index % 2 == 0 else "MG"
                profile = [20, 2, 4, 10, 3, 3] if cell_type == "Rod" else [2, 20, 4, 10, 3, 3]
                row = rng.poisson(profile).astype(int).tolist()
                spatial = assay == "xenium"
                cells.append({
                    "cell_id": cell_id, "record_id": record_id, "cell_type": cell_type,
                    "layer": f"SYN_LAYER_{cell_index // 8}" if spatial else "",
                    "x": float((cell_index % 8) * 10 + 5) if spatial else "",
                    "y": float((cell_index // 8) * 15 + 5) if spatial else "",
                    "tissue_component": "SYN_COMPONENT_A" if spatial else "",
                })
                counts.append([cell_id, *row])
    with (path / "cells.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_id", "record_id", "cell_type", "layer", "x", "y", "tissue_component"])
        writer.writeheader()
        writer.writerows(cells)
    # Deliberately change row order: readers must align cell IDs explicitly.
    with (path / "counts.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", *GENES])
        writer.writerows(counts[::-1])
    manifest = {
        "schema_version": 1, "data_kind": "synthetic", "dataset_id": prefix,
        "source_version": "synthetic_generator_v1", "counts_file": "counts.csv",
        "cells_file": "cells.csv", "records": records,
    }
    write_json(path / "manifest.json", manifest)
    return path / "manifest.json"


def generate_synthetic_fixtures(output_dir: str | Path, *, seed: int = 11) -> dict[str, Path]:
    """Create a new directory; never reuse or overwrite an existing fixture set."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    rng = np.random.default_rng(seed)
    manifests = {
        "rna": _write_cohort(destination / "rna", "chromium_snrna", rng),
        "spatial": _write_cohort(destination / "spatial", "xenium", rng),
    }
    write_json(destination / "generator.json", {
        "data_kind": "synthetic", "generator": "synthetic_generator_v1", "seed": seed,
        "independent_disease_evidence": False,
        "purpose": "pipeline behavior checks; no measured human biology",
    })
    return manifests
