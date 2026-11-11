"""Deterministic spatial descriptors for verified physical tissue sections.

Coordinates and ``radius_um`` use micrometres. Graphs are directed: each source
selects at most ``max_neighbors`` targets. No function fits a disease model.
Conditional residuals describe a specified permutation null, not a population
abnormality distribution. NumPy arrays in returned dictionaries require explicit
conversion before JSON serialization.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral
from typing import Any

import numpy as np


_MISSING = {"", "unknown", "unresolved", "conflicting", "none", "null", "not_applicable", "restricted"}
_EPS = 1e-9


def _known_key(value: Any, what: str) -> str | tuple:
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"{what} must be explicit")
        return tuple(_known_key(item, what) for item in value)
    if not isinstance(value, str) or value.strip().lower() in _MISSING:
        raise ValueError(f"{what} must be explicit")
    return value


def _keys(values: Sequence, n: int, what: str) -> list:
    if len(values) != n:
        raise ValueError(f"{what} length does not match cells")
    return [_known_key(value, what) for value in values]


def _positive_int(value: Any, what: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{what} must be an integer at least {minimum}")
    return int(value)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    length = math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
    if abs(_cross(b - a, p - a)) > _EPS * max(1.0, length):
        return False
    return bool(np.all(p >= np.minimum(a, b) - _EPS) and np.all(p <= np.maximum(a, b) + _EPS))


def _point_location(point: np.ndarray, polygon: np.ndarray) -> int:
    """Return -1 outside, 0 on the boundary, and 1 strictly inside."""
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if _on_segment(point, a, b):
            return 0
        if (a[1] > point[1]) != (b[1] > point[1]):
            crossing_x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < crossing_x:
                inside = not inside
    return 1 if inside else -1


def _intersection_parameters(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> list[float]:
    """Parameters on segment ab that intersect cd, including collinear overlap."""
    direction = b - a
    other = d - c
    denominator = _cross(direction, other)
    tolerance = _EPS * max(1.0, math.hypot(*direction), math.hypot(*other))
    if abs(denominator) <= tolerance:
        if abs(_cross(c - a, direction)) > tolerance:
            return []
        squared_length = float(np.dot(direction, direction))
        if squared_length == 0:
            return [0.0] if _on_segment(a, c, d) else []
        lower, upper = sorted((float(np.dot(c - a, direction) / squared_length), float(np.dot(d - a, direction) / squared_length)))
        if upper < -_EPS or lower > 1 + _EPS:
            return []
        return [max(0.0, lower), min(1.0, upper)]
    t = _cross(c - a, other) / denominator
    u = _cross(c - a, direction) / denominator
    if -_EPS <= t <= 1 + _EPS and -_EPS <= u <= 1 + _EPS:
        return [min(1.0, max(0.0, t))]
    return []


def _polygon(raw: Any) -> np.ndarray:
    try:
        polygon = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("tissue polygon must contain finite coordinates") from exc
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        raise ValueError("tissue polygon must have shape (vertices, 2) and finite coordinates")
    if len(polygon) > 1 and np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if len(polygon) < 3 or len(np.unique(polygon, axis=0)) != len(polygon):
        raise ValueError("tissue polygon needs at least three distinct vertices")
    shifted = polygon - polygon[0]
    area = sum(_cross(shifted[i], shifted[(i + 1) % len(shifted)]) for i in range(len(shifted)))
    if not math.isfinite(area) or abs(area) <= _EPS:
        raise ValueError("tissue polygon area is invalid")
    for i in range(len(polygon)):
        for j in range(i + 1, len(polygon)):
            if j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            if _intersection_parameters(polygon[i], polygon[(i + 1) % len(polygon)], polygon[j], polygon[(j + 1) % len(polygon)]):
                raise ValueError("tissue polygon is self-intersecting")
    return polygon


def _prepare_mask(raw: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    if not isinstance(raw, Mapping) or "outer" not in raw:
        raise ValueError("tissue mask needs an outer polygon")
    outer = _polygon(raw["outer"])
    raw_holes = raw.get("holes", [])
    if not isinstance(raw_holes, (list, tuple)):
        raise ValueError("tissue mask holes must be a sequence of polygons")
    holes = [_polygon(hole) for hole in raw_holes]
    for hole in holes:
        if any(_point_location(point, outer) != 1 for point in hole):
            raise ValueError("tissue holes must lie strictly inside the outer polygon")
        for a, b in zip(hole, np.roll(hole, -1, axis=0)):
            if any(_intersection_parameters(a, b, c, d) for c, d in zip(outer, np.roll(outer, -1, axis=0))):
                raise ValueError("tissue hole crosses the outer boundary")
    for i, hole in enumerate(holes):
        for other in holes[i + 1:]:
            if _point_location(hole[0], other) >= 0 or _point_location(other[0], hole) >= 0:
                raise ValueError("tissue holes must not overlap")
            if any(_intersection_parameters(a, b, c, d) for a, b in zip(hole, np.roll(hole, -1, axis=0)) for c, d in zip(other, np.roll(other, -1, axis=0))):
                raise ValueError("tissue holes must not overlap")
    return outer, holes


def _inside_mask(point: np.ndarray, mask: tuple[np.ndarray, list[np.ndarray]]) -> bool:
    outer, holes = mask
    return _point_location(point, outer) >= 0 and all(_point_location(point, hole) < 0 for hole in holes)


def _segment_in_mask(a: np.ndarray, b: np.ndarray, mask: tuple[np.ndarray, list[np.ndarray]]) -> bool:
    outer, holes = mask
    for hole in holes:
        if any(_intersection_parameters(a, b, c, d) for c, d in zip(hole, np.roll(hole, -1, axis=0))):
            return False
    parameters = [0.0, 1.0]
    for c, d in zip(outer, np.roll(outer, -1, axis=0)):
        parameters.extend(_intersection_parameters(a, b, c, d))
    parameters = sorted(set(parameters))
    for left, right in zip(parameters, parameters[1:]):
        midpoint = a + ((left + right) / 2) * (b - a)
        if _point_location(midpoint, outer) < 0:
            return False
    return True


def build_radius_graph(coordinates, section_keys, *, radius_um, max_neighbors, cell_ids=None, tissue_components=None, tissue_masks=None) -> np.ndarray:
    """Build a directed, capped radius graph using section-local spatial grids.

    Mask outer boundaries are included; hole boundaries are excluded. Nodes
    outside supplied tissue raise an error instead of being silently dropped.
    Every represented section needs a mask when ``tissue_masks`` is supplied.
    Without masks, this function establishes section/component locality only.
    """
    try:
        coordinates = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be finite micrometre values") from exc
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must have shape (cells, 2) and be finite")
    if isinstance(radius_um, (bool, np.bool_)):
        raise ValueError("radius_um must be positive and finite")
    try:
        radius_um = float(radius_um)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius_um must be positive and finite") from exc
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be positive and finite")
    max_neighbors = _positive_int(max_neighbors, "max_neighbors")
    n = len(coordinates)
    sections = _keys(section_keys, n, "section key")
    components = _keys(tissue_components, n, "tissue component") if tissue_components is not None else [None] * n
    if cell_ids is None:
        ids = [str(index) for index in range(n)]
    else:
        ids = _keys(cell_ids, n, "cell ID")
        if any(not isinstance(value, str) for value in ids) or len(set(ids)) != n:
            raise ValueError("cell IDs must be unique strings")
    masks = None
    if tissue_masks is not None:
        if not isinstance(tissue_masks, Mapping):
            raise ValueError("tissue_masks must map section keys to polygons")
        masks = {_known_key(key, "mask section key"): _prepare_mask(value) for key, value in tissue_masks.items()}
        if any(section not in masks for section in sections):
            raise ValueError("a supplied tissue mask is missing a section")
        if any(not _inside_mask(coordinates[i], masks[sections[i]]) for i in range(n)):
            raise ValueError("a cell lies outside valid masked tissue")
    grids: dict[tuple, list[int]] = defaultdict(list)
    bins = []
    for i, (x, y) in enumerate(coordinates):
        try:
            location = (math.floor(float(x) / radius_um), math.floor(float(y) / radius_um))
        except (OverflowError, ValueError) as exc:
            raise ValueError("coordinate/radius scale cannot be indexed safely") from exc
        bins.append(location)
        grids[(sections[i], components[i], *location)].append(i)
    sources, targets = [], []
    for i, (gx, gy) in enumerate(bins):
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grids.get((sections[i], components[i], gx + dx, gy + dy), ()):
                    if i == j:
                        continue
                    distance = math.hypot(float(coordinates[j, 0]) - float(coordinates[i, 0]), float(coordinates[j, 1]) - float(coordinates[i, 1]))
                    if distance <= radius_um and (masks is None or _segment_in_mask(coordinates[i], coordinates[j], masks[sections[i]])):
                        candidates.append((distance, ids[j], j))
        for _, _, j in sorted(candidates)[:max_neighbors]:
            sources.append(i)
            targets.append(j)
    return np.asarray([sources, targets], dtype=np.int64)


def _labels(values) -> list[str | None]:
    if isinstance(values, (str, bytes)):
        raise ValueError("labels must be a sequence")
    result = []
    for value in values:
        if value is None or (isinstance(value, str) and value.strip().lower() in _MISSING):
            result.append(None)
        elif isinstance(value, str):
            result.append(value)
        else:
            raise ValueError("labels must be strings or explicit missing values")
    return result


def _validated_edges(edges, n: int) -> np.ndarray:
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edges must be an integer array with shape (2, E)")
    if edges.size and (np.any(edges < 0) or np.any(edges >= n)):
        raise ValueError("edge endpoint is outside the cell array")
    if edges.shape[1] and np.any(edges[0] == edges[1]):
        raise ValueError("self-edges are not permitted")
    if len(set(zip(edges[0].tolist(), edges[1].tolist()))) != edges.shape[1]:
        raise ValueError("duplicate edges are not permitted")
    return edges.astype(np.int64, copy=False)


def neighborhood_fractions(edges, labels, *, categories=None) -> dict:
    """Count target-cell labels in each source's directed neighborhood."""
    labels = _labels(labels)
    n = len(labels)
    edges = _validated_edges(edges, n)
    if categories is None:
        categories = sorted({label for label in labels if label is not None})
    else:
        categories = _labels(categories)
        if len(set(categories)) != len(categories):
            raise ValueError("categories must be unique")
        if any(label is not None and label not in categories for label in labels):
            raise ValueError("categories omit an observed label")
    if None in labels and None not in categories:
        categories.append(None)
    category_index = {label: index for index, label in enumerate(categories)}
    counts = np.zeros((n, len(categories)), dtype=np.int64)
    if edges.shape[1]:
        target_categories = np.asarray([category_index[labels[index]] for index in edges[1]], dtype=np.int64)
        np.add.at(counts, (edges[0], target_categories), 1)
    degree = counts.sum(axis=1)
    fractions = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, degree[:, None], out=fractions, where=degree[:, None] != 0)
    unknown = counts[:, category_index[None]].copy() if None in category_index else np.zeros(n, dtype=np.int64)
    return {"categories": list(categories), "counts": counts, "fractions": fractions, "degree": degree, "is_isolate": degree == 0, "unknown_label_count": unknown, "n_unknown_cells": sum(label is None for label in labels)}


def stratified_permutation(labels, strata, *, seed) -> np.ndarray:
    """Permute labels within explicit strata, retaining each stratum's counts.

    Each stratum receives a stable seed derived from its key and the run seed.
    Adding another stratum cannot alter an existing stratum's permutation.
    """
    labels = _labels(labels)
    strata = _keys(strata, len(labels), "permutation stratum")
    seed = _positive_int(seed, "seed", 0)
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(strata):
        groups[key].append(index)
    result = np.asarray(labels, dtype=object)
    for key, indices in groups.items():
        seed_bytes = json.dumps([seed, key], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        stable_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:16], "little")
        generator = np.random.default_rng(stable_seed)
        result[indices] = generator.permutation(result[indices])
    return result


def conditional_neighborhood_residuals(edges, labels, strata, *, n_permutations, seed, categories=None) -> dict:
    """Compare observed fractions to a seeded stratum-preserving label null.

    Singleton strata cannot change. Zero null variance is explicit, with finite
    zero standardized values; raw residuals remain available for interpretation.
    The caller must define strata that contain a physical section identifier.
    """
    n_permutations = _positive_int(n_permutations, "n_permutations", 2)
    seed = _positive_int(seed, "seed", 0)
    labels = _labels(labels)
    strata = _keys(strata, len(labels), "permutation stratum")
    observed = neighborhood_fractions(edges, labels, categories=categories)
    fixed_categories = observed["categories"]
    mean = np.zeros_like(observed["fractions"])
    m2 = np.zeros_like(mean)
    generator = np.random.default_rng(seed)
    for iteration in range(1, n_permutations + 1):
        permutation_seed = int(generator.integers(0, np.iinfo(np.int64).max))
        permuted = stratified_permutation(labels, strata, seed=permutation_seed)
        value = neighborhood_fractions(edges, permuted, categories=fixed_categories)["fractions"]
        delta = value - mean
        mean += delta / iteration
        m2 += delta * (value - mean)
    sd = np.sqrt(np.maximum(0.0, m2 / (n_permutations - 1)))
    residual = observed["fractions"] - mean
    zero_variance = sd <= np.finfo(np.float64).eps
    standardized = np.zeros_like(residual)
    np.divide(residual, sd, out=standardized, where=~zero_variance)
    return {"categories": fixed_categories, "observed": observed["fractions"], "null_mean": mean, "null_sd": sd, "residual": residual, "standardized_residual": standardized, "zero_variance": zero_variance, "is_isolate": observed["is_isolate"], "n_permutations": n_permutations, "seed": seed, "interpretation": "conditional_spatial_null_descriptor"}


def _json_key(value):
    return [_json_key(item) for item in value] if isinstance(value, tuple) else value


def pool_sections_to_donors(values, donor_ids, section_ids, regions) -> list[dict]:
    """Mean cells within sections, then give each section equal donor weight.

    Retinal regions remain separate. A donor can have identically named local
    sections in a different donor; composite section IDs are accepted. Means are
    always lists, including for a single scalar feature.
    """
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pooling values must be finite numerical features") from exc
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("pooling values must be a finite cells-by-features array")
    n = len(values)
    donors = _keys(donor_ids, n, "donor ID")
    sections = _keys(section_ids, n, "section ID")
    regions = _keys(regions, n, "retinal region")
    if any(not isinstance(key, str) for key in donors + regions):
        raise ValueError("donor IDs and retinal regions must be strings")
    groups: dict[tuple, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (donor, section, region) in enumerate(zip(donors, sections, regions)):
        groups[(donor, region)][section].append(index)
    rows = []
    for (donor, region), section_groups in sorted(groups.items()):
        ordered_sections = sorted(section_groups, key=repr)
        means = [values[section_groups[section]].mean(axis=0) for section in ordered_sections]
        rows.append({"donor_id": donor, "retinal_region": region, "n_sections": len(ordered_sections), "n_cells": sum(len(indices) for indices in section_groups.values()), "section_ids": [_json_key(section) for section in ordered_sections], "mean": np.asarray(means).mean(axis=0).tolist()})
    return rows
