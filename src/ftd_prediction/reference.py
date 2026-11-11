"""A small frozen cell-annotation reference baseline, with no disease head."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import ContractError, is_resolved, normalize_annotation
from .data import _unique_json_object


def _names(values: Iterable, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ContractError(f"The {field} must be a sequence.")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ContractError(f"The {field} must be a sequence.") from error
    for value in result:
        if not isinstance(value, str) or not value or value != value.strip() or any(ord(char) < 32 for char in value):
            raise ContractError(f"The {field} contains a missing or malformed value.")
    return result


def _genes(values: Iterable) -> tuple[str, ...]:
    genes = _names(values, "gene vocabulary")
    if not genes or len(set(genes)) != len(genes):
        raise ContractError("Reference gene vocabulary must be nonempty and unique.")
    return genes


def _counts(values, n_genes: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != n_genes or array.dtype.kind not in "iuf":
        raise ContractError("Reference counts must be a numeric cells-by-genes matrix.")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ContractError("Reference counts must be finite and nonnegative.")
    if array.dtype.kind == "f":
        if np.any(array != np.floor(array)) or np.any(array > 2**53):
            raise ContractError("Floating raw counts must be exact integers within the supported range.")
    elif array.dtype.kind == "u" and np.any(array > np.uint64(np.iinfo(np.int64).max)):
        raise ContractError("Reference counts exceed the supported integer range.")
    return array.astype(np.float64, copy=True)


def _normalize(counts: np.ndarray, total: float) -> tuple[np.ndarray, np.ndarray]:
    sums = counts.sum(axis=1)
    normalized = np.zeros_like(counts)
    nonzero = sums > 0
    normalized[nonzero] = np.log1p(counts[nonzero] / sums[nonzero, None] * total)
    return normalized, nonzero


@dataclass(frozen=True)
class CentroidReference:
    """Frozen donor-balanced centroids in library-normalized log expression.

    This baseline predicts cell-type labels, not donor diagnosis. Each cell type
    is averaged within each fitting donor, then equally across those donors.
    The gene vocabulary is supplied explicitly and no query-wide fitting occurs.
    """

    genes: tuple[str, ...]
    categories: tuple[str, ...]
    centroids: np.ndarray
    normalization_total: float
    provenance: dict

    @classmethod
    def fit(
        cls, counts, genes, cell_types, donor_ids, *, fit_donors,
        normalization_total: float = 10_000.0,
    ) -> "CentroidReference":
        """Fit only explicitly allowed donors, recording every exclusion.

        Rows from other donors are excluded before count or label-derived
        parameters are calculated. Selected rows with missing annotations are
        excluded explicitly. No label or donor information is inferred.
        """
        vocabulary = _genes(genes)
        donors = _names(donor_ids, "donor identifiers")
        allowed = _names(fit_donors, "allowed fitting donors")
        if not allowed or len(set(allowed)) != len(allowed) or any(not is_resolved(donor) for donor in allowed):
            raise ContractError("Allowed fitting donors must be explicit, unique and nonempty.")
        if isinstance(cell_types, (str, bytes)):
            raise ContractError("Cell annotations must be a sequence.")
        try:
            labels = tuple(cell_types)
        except TypeError as error:
            raise ContractError("Cell annotations must be a sequence.") from error
        matrix = np.asarray(counts)
        if matrix.ndim != 2 or matrix.shape != (len(donors), len(vocabulary)) or len(labels) != len(donors):
            raise ContractError("Fitting metadata and count matrix shapes disagree.")
        if isinstance(normalization_total, bool) or not isinstance(normalization_total, (int, float)):
            raise ContractError("The normalization total must be positive and finite.")
        total = float(normalization_total)
        if not math.isfinite(total) or total <= 0:
            raise ContractError("The normalization total must be positive and finite.")
        eligible = []
        selected_labels = []
        missing_annotations = 0
        allowed_set = set(allowed)
        for index, (donor, label) in enumerate(zip(donors, labels)):
            if donor not in allowed_set:
                continue
            normalized_label = normalize_annotation(label)
            if normalized_label is None:
                missing_annotations += 1
                continue
            eligible.append(index)
            selected_labels.append(normalized_label)
        if not eligible:
            raise ContractError("No labelled cells belong to the allowed fitting donors.")
        training_counts = _counts(matrix[eligible], len(vocabulary))
        normalized, informative = _normalize(training_counts, total)
        if not np.all(informative):
            raise ContractError("Labelled fitting cells must have nonzero libraries.")
        training_labels = tuple(selected_labels)
        training_donors = tuple(donors[index] for index in eligible)
        categories = tuple(sorted(set(training_labels)))
        centroids = []
        for category in categories:
            donor_means = []
            for donor in sorted(set(training_donors)):
                selection = [index for index, pair in enumerate(zip(training_labels, training_donors)) if pair == (category, donor)]
                if selection:
                    donor_means.append(normalized[selection].mean(axis=0))
            centroids.append(np.mean(donor_means, axis=0))
        centers = np.asarray(centroids, dtype=np.float64)
        centers.setflags(write=False)
        excluded = sum(donor not in allowed_set for donor in donors)
        provenance = {
            "schema_version": 1,
            "model_type": "donor_balanced_cell_type_centroid",
            "evidence_scope": "cell_annotation_reference",
            "requested_fit_donors": sorted(allowed_set),
            "fit_donors": sorted(set(training_donors)),
            "n_input_rows": len(donors),
            "n_fitted_rows": len(eligible),
            "n_excluded_donor_rows": int(excluded),
            "n_missing_annotation_rows": int(missing_annotations),
            "normalization": "per_cell_library_size_log1p",
            "normalization_total": total,
            "pooling": "equal_mean_of_donor_means_within_cell_type",
        }
        return cls(vocabulary, categories, centers, total, provenance)

    def predict(self, counts, genes) -> dict:
        """Align genes explicitly and map each query cell independently.

        Extra genes are ignored; all reference genes must be supplied. Returned
        distances are Euclidean distances and scores are their negative squares.
        They are similarities, not calibrated cell or disease probabilities.
        Empty-library cells have a missing label and an explicit quality flag.
        """
        query_genes = _genes(genes)
        positions = {gene: index for index, gene in enumerate(query_genes)}
        if not set(self.genes).issubset(positions):
            raise ContractError("The query is missing required reference genes.")
        matrix = _counts(counts, len(query_genes))
        aligned = matrix[:, [positions[gene] for gene in self.genes]]
        normalized, informative = _normalize(aligned, self.normalization_total)
        # Avoid constructing a cells-by-classes-by-genes temporary array.
        squared = np.empty((len(normalized), len(self.categories)), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for index, center in enumerate(self.centroids):
                difference = normalized - center
                squared[:, index] = np.einsum("ij,ij->i", difference, difference)
        if not np.all(np.isfinite(squared)):
            raise ContractError("Reference inference produced nonfinite distances.")
        labels = [self.categories[int(index)] if informative[row] else None
                  for row, index in enumerate(np.argmin(squared, axis=1))]
        return {
            "labels": labels,
            "categories": list(self.categories),
            "distances": np.sqrt(squared).tolist(),
            "scores": (-squared).tolist(),
            "zero_library": (~informative).tolist(),
            "evidence_scope": "cell_annotation_reference",
            "score_definition": "negative_squared_euclidean_distance_not_probability",
        }

    def save(self, path) -> None:
        """Save a new versioned JSON artifact, never replacing an existing file."""
        payload = {
            "schema_version": 1,
            "model_type": "donor_balanced_cell_type_centroid",
            "genes": list(self.genes),
            "categories": list(self.categories),
            "centroids": self.centroids.tolist(),
            "normalization_total": self.normalization_total,
            "provenance": self.provenance,
        }
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path) -> "CentroidReference":
        """Load the constrained JSON artifact without pickle or executable data."""
        source = Path(path)
        try:
            if source.stat().st_size > 32 * 1024 * 1024:
                raise ContractError("The reference model exceeds the supported artifact size.")
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_unique_json_object)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError("The reference model artifact could not be read.") from error
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ContractError("Unsupported reference model schema.")
        if payload.get("model_type") != "donor_balanced_cell_type_centroid":
            raise ContractError("Unsupported reference model type.")
        genes = _genes(payload.get("genes"))
        categories = _names(payload.get("categories"), "cell-type categories")
        if not categories or len(categories) != len(set(categories)):
            raise ContractError("Reference cell-type categories must be nonempty and unique.")
        if any(normalize_annotation(category) is None for category in categories):
            raise ContractError("Reference cell-type categories cannot contain missing-value tokens.")
        if isinstance(payload.get("normalization_total"), bool) or not isinstance(payload.get("normalization_total"), (int, float)):
            raise ContractError("The reference normalization total is malformed.")
        try:
            centers = np.asarray(payload.get("centroids"), dtype=np.float64)
            total = float(payload.get("normalization_total"))
        except (TypeError, ValueError, OverflowError) as error:
            raise ContractError("The reference model parameters are malformed.") from error
        if centers.shape != (len(categories), len(genes)) or not np.all(np.isfinite(centers)) or np.any(centers < 0):
            raise ContractError("The reference model centroids are malformed.")
        if not math.isfinite(total) or total <= 0:
            raise ContractError("The reference normalization total is malformed.")
        bound = math.log1p(total)
        if np.any(centers > bound + 1e-10 * max(1.0, bound)):
            raise ContractError("The reference centroids exceed the normalization bounds.")
        provenance = payload.get("provenance")
        required_provenance = {
            "schema_version", "model_type", "evidence_scope", "requested_fit_donors",
            "fit_donors", "n_input_rows", "n_fitted_rows", "n_excluded_donor_rows",
            "n_missing_annotation_rows", "normalization", "normalization_total", "pooling",
        }
        if not isinstance(provenance, dict) or not required_provenance.issubset(provenance):
            raise ContractError("The reference provenance is missing or unsupported.")
        expected_provenance = {
            "model_type": "donor_balanced_cell_type_centroid",
            "evidence_scope": "cell_annotation_reference",
            "normalization": "per_cell_library_size_log1p",
            "pooling": "equal_mean_of_donor_means_within_cell_type",
        }
        if type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1:
            raise ContractError("The reference provenance version is unsupported.")
        if any(provenance[key] != expected for key, expected in expected_provenance.items()):
            raise ContractError("The reference provenance is inconsistent with its model type.")
        provenance_total = provenance["normalization_total"]
        if isinstance(provenance_total, bool) or not isinstance(provenance_total, (int, float)) or provenance_total != total:
            raise ContractError("The reference normalization and provenance disagree.")
        fitting_donors = _names(provenance.get("fit_donors"), "fitting donor provenance")
        if not fitting_donors or len(fitting_donors) != len(set(fitting_donors)) or any(not is_resolved(donor) for donor in fitting_donors):
            raise ContractError("The reference fitting-donor provenance is malformed.")
        requested = _names(provenance["requested_fit_donors"], "requested fitting donor provenance")
        if not requested or len(requested) != len(set(requested)) or any(not is_resolved(donor) for donor in requested):
            raise ContractError("The requested fitting-donor provenance is malformed.")
        if not set(fitting_donors).issubset(requested):
            raise ContractError("The fitted donors are not a subset of the requested donors.")
        for field in ("n_input_rows", "n_fitted_rows", "n_excluded_donor_rows", "n_missing_annotation_rows"):
            if type(provenance[field]) is not int or provenance[field] < 0:
                raise ContractError("The reference row-count provenance is malformed.")
        if provenance["n_fitted_rows"] <= 0 or provenance["n_input_rows"] != sum(provenance[field] for field in ("n_fitted_rows", "n_excluded_donor_rows", "n_missing_annotation_rows")):
            raise ContractError("The reference row-count provenance is inconsistent.")
        if len(fitting_donors) > provenance["n_fitted_rows"] or len(categories) > provenance["n_fitted_rows"]:
            raise ContractError("The fitted categories or donors exceed the recorded fitting rows.")
        centers.setflags(write=False)
        return cls(genes, categories, centers, total, provenance)
