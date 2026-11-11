"""Independent synthetic checks for section-local spatial descriptors."""

import collections
import json
import unittest

import numpy as np

from ftd_prediction.spatial import (
    build_radius_graph,
    conditional_neighborhood_residuals,
    neighborhood_fractions,
    pool_sections_to_donors,
    stratified_permutation,
)


class RadiusGraphTests(unittest.TestCase):
    def test_overlapping_coordinates_cannot_cross_sections_or_components(self):
        xy = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0.5, 0]])
        keys = [("d1", "sample", "s1")] * 2 + [("d2", "sample", "s1")] * 2 + [("d1", "sample", "s1")]
        edges = build_radius_graph(xy, keys, radius_um=5, max_neighbors=4, tissue_components=["a", "a", "a", "a", "b"])
        self.assertEqual(set(map(tuple, edges.T)), {(0, 1), (1, 0), (2, 3), (3, 2)})

    def test_cap_and_cell_id_tie_breaking_survive_input_reordering(self):
        xy = np.array([[0, 0], [-1, 0], [1, 0], [0, 1]])
        ids = ["origin", "z", "a", "b"]
        edges = build_radius_graph(xy, ["s"] * 4, radius_um=2, max_neighbors=1, cell_ids=ids)
        self.assertEqual(edges[:, 0].tolist(), [0, 2])
        self.assertLessEqual(collections.Counter(edges[0]).most_common(1)[0][1], 1)
        reorder = [3, 0, 1, 2]
        shifted = build_radius_graph(xy[reorder], ["s"] * 4, radius_um=2, max_neighbors=1, cell_ids=[ids[i] for i in reorder])
        original_named = {(ids[a], ids[b]) for a, b in edges.T}
        shifted_named = {(ids[reorder[a]], ids[reorder[b]]) for a, b in shifted.T}
        self.assertEqual(original_named, shifted_named)

    def test_grid_matches_independent_brute_force(self):
        rng = np.random.default_rng(29)
        xy = rng.normal(size=(73, 2)) * 20
        sections = ["a" if i % 3 else "b" for i in range(len(xy))]
        ids = [f"cell{i:03}" for i in range(len(xy))]
        edges = build_radius_graph(xy, sections, radius_um=8, max_neighbors=5, cell_ids=ids)
        expected = []
        for i in range(len(xy)):
            candidates = [(float(np.linalg.norm(xy[j] - xy[i])), ids[j], j) for j in range(len(xy)) if i != j and sections[i] == sections[j] and np.linalg.norm(xy[j] - xy[i]) <= 8]
            expected.extend((i, j) for _, _, j in sorted(candidates)[:5])
        self.assertEqual(list(map(tuple, edges.T)), expected)

    def test_radius_boundary_and_rigid_transform(self):
        xy = np.array([[0., 0.], [3., 4.], [10., 0.]])
        before = xy.copy()
        edges = build_radius_graph(xy, ["s"] * 3, radius_um=5, max_neighbors=3)
        self.assertEqual(set(map(tuple, edges.T)), {(0, 1), (1, 0)})
        rotation = np.array([[0., -1.], [1., 0.]])
        transformed = xy @ rotation + [120, -50]
        np.testing.assert_array_equal(edges, build_radius_graph(transformed, ["s"] * 3, radius_um=5, max_neighbors=3))
        np.testing.assert_array_equal(xy, before)

    def test_hole_crossing_and_tangency_are_excluded(self):
        mask = {"s": {"outer": [[0, 0], [10, 0], [10, 10], [0, 10]], "holes": [[[4, 4], [6, 4], [6, 6], [4, 6]]]}}
        xy = np.array([[2, 5], [8, 5], [2, 2], [8, 2], [2, 4], [8, 4]], dtype=float)
        edges = set(map(tuple, build_radius_graph(xy, ["s"] * 6, radius_um=7, max_neighbors=10, tissue_masks=mask).T))
        self.assertNotIn((0, 1), edges)
        self.assertNotIn((4, 5), edges)
        self.assertIn((2, 3), edges)

    def test_concave_outer_boundary_rejects_crossing_segment(self):
        mask = {"s": {"outer": [[0, 0], [6, 0], [6, 6], [4, 6], [4, 2], [2, 2], [2, 6], [0, 6]], "holes": []}}
        edges = build_radius_graph([[1, 5], [5, 5], [1, 1]], ["s"] * 3, radius_um=6, max_neighbors=3, tissue_masks=mask)
        self.assertNotIn((0, 1), set(map(tuple, edges.T)))
        self.assertIn((0, 2), set(map(tuple, edges.T)))

    def test_mask_nodes_and_coverage_fail_closed(self):
        mask = {"s": {"outer": [[0, 0], [2, 0], [2, 2], [0, 2]]}}
        with self.assertRaises(ValueError):
            build_radius_graph([[3, 3]], ["s"], radius_um=1, max_neighbors=1, tissue_masks=mask)
        with self.assertRaises(ValueError):
            build_radius_graph([[1, 1]], ["other"], radius_um=1, max_neighbors=1, tissue_masks=mask)
        hole_mask = {"s": {"outer": [[0, 0], [4, 0], [4, 4], [0, 4]], "holes": [[[1, 1], [2, 1], [2, 2], [1, 2]]]}}
        with self.assertRaises(ValueError):
            build_radius_graph([[1, 1]], ["s"], radius_um=1, max_neighbors=1, tissue_masks=hole_mask)

    def test_invalid_geometry_and_arguments(self):
        cases = [({"coordinates": [[float("nan"), 0]], "section_keys": ["s"]}), ({"coordinates": [[0, 0]], "section_keys": ["unknown"]}), ({"coordinates": [[0, 0]], "section_keys": [("d", "sample", None)]})]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_radius_graph(**kwargs, radius_um=1, max_neighbors=1)
        for radius in [0, -1, float("inf"), True]:
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                build_radius_graph([[0, 0]], ["s"], radius_um=radius, max_neighbors=1)
        with self.assertRaises(ValueError):
            build_radius_graph([[0, 0]], ["s"], radius_um=1, max_neighbors=0)
        with self.assertRaises(ValueError):
            build_radius_graph([[0, 0], [1, 0]], ["s", "s"], radius_um=1, max_neighbors=1, cell_ids=["duplicate", "duplicate"])
        with self.assertRaises(ValueError):
            build_radius_graph([[0, 0]], ["s"], radius_um=1, max_neighbors=1, tissue_components=[None])
        with self.assertRaises(ValueError):
            build_radius_graph([[0, 0]], ["s"], radius_um=1, max_neighbors=1, tissue_masks={"s": {"outer": [[0, 0], [2, 2], [0, 2], [2, 0]]}})

    def test_empty_graph_has_stable_shape(self):
        result = build_radius_graph(np.empty((0, 2)), [], radius_um=1, max_neighbors=2)
        self.assertEqual(result.shape, (2, 0))
        self.assertEqual(result.dtype, np.int64)


class NeighborhoodTests(unittest.TestCase):
    def test_fractions_isolates_and_unknown_labels(self):
        edges = np.array([[0, 0, 1], [1, 2, 0]])
        result = neighborhood_fractions(edges, ["A", "B", None, "A"])
        self.assertEqual(result["categories"], ["A", "B", None])
        np.testing.assert_array_equal(result["counts"], [[0, 1, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(result["fractions"][0], [0, .5, .5])
        self.assertEqual(result["is_isolate"].tolist(), [False, False, True, True])
        self.assertEqual(result["unknown_label_count"].tolist(), [1, 0, 0, 0])
        self.assertEqual(result["n_unknown_cells"], 1)

    def test_categories_and_bad_edges_are_validated(self):
        for edges in [np.array([[0], [0]]), np.array([[0, 0], [1, 1]]), np.array([[0], [2]]), np.array([[0.], [1.]])]:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                neighborhood_fractions(edges, ["A", "B"])
        with self.assertRaises(ValueError):
            neighborhood_fractions(np.empty((2, 0), dtype=int), ["A", "B"], categories=["A"])

    def test_permutation_preserves_each_stratum_and_inputs(self):
        labels = ["A", "B", "C", None, "A", "B", "C"]
        strata = [("s1", "L1")] * 3 + [("s1", "L2")] * 2 + [("s2", "L1")] * 2
        before = list(labels)
        first = stratified_permutation(labels, strata, seed=47)
        second = stratified_permutation(labels, strata, seed=47)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(labels, before)
        for stratum in set(strata):
            indices = [i for i, key in enumerate(strata) if key == stratum]
            self.assertEqual(collections.Counter(labels[i] for i in indices), collections.Counter(first[i] for i in indices))
        with self.assertRaises(ValueError):
            stratified_permutation(["A"], [("s", None)], seed=1)

    def test_conditional_null_is_reproducible_and_finite(self):
        edges = np.array([[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 1, 1]])
        labels = ["A", "A", "B", "B", "C"]
        strata = [("s", "layer")] * 4 + [("s", "isolated")]
        result = conditional_neighborhood_residuals(edges, labels, strata, n_permutations=32, seed=11)
        repeated = conditional_neighborhood_residuals(edges, labels, strata, n_permutations=32, seed=11)
        for key in ["observed", "null_mean", "null_sd", "residual", "standardized_residual"]:
            self.assertTrue(np.isfinite(result[key]).all())
            np.testing.assert_array_equal(result[key], repeated[key])
        np.testing.assert_allclose(result["residual"], result["observed"] - result["null_mean"])
        self.assertTrue(result["is_isolate"][4])
        self.assertTrue(result["zero_variance"][4].all())
        self.assertEqual(result["interpretation"], "conditional_spatial_null_descriptor")

    def test_singleton_strata_expose_degenerate_null(self):
        edges = np.array([[0, 1], [1, 0]])
        result = conditional_neighborhood_residuals(edges, ["A", "B"], ["stratum1", "stratum2"], n_permutations=3, seed=0)
        self.assertTrue(result["zero_variance"].all())
        np.testing.assert_array_equal(result["residual"], np.zeros((2, 2)))
        np.testing.assert_array_equal(result["standardized_residual"], np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            conditional_neighborhood_residuals(edges, ["A", "B"], ["s", "s"], n_permutations=1, seed=0)

    def test_unrelated_sections_do_not_change_existing_null(self):
        edges = np.array([[0, 0, 1, 2], [1, 2, 2, 0]])
        labels = ["A", "B", "B"]
        strata = [("s1", "layer")] * 3
        original = conditional_neighborhood_residuals(edges, labels, strata, n_permutations=19, seed=29)
        # Insert an unrelated section before the original, exercising random
        # stream independence rather than merely appending an unused group.
        combined_edges = np.concatenate((np.array([[0, 1], [1, 0]]), edges + 2), axis=1)
        combined = conditional_neighborhood_residuals(combined_edges, ["A", "B"] + labels, [("new", "layer")] * 2 + strata, n_permutations=19, seed=29)
        for key in ["observed", "null_mean", "null_sd", "residual", "standardized_residual"]:
            np.testing.assert_array_equal(original[key], combined[key][2:])


class PoolingTests(unittest.TestCase):
    def test_sections_have_equal_weight_and_regions_remain_distinct(self):
        values = [[0.], [0.], [0.], [10.], [100.], [5.]]
        rows = pool_sections_to_donors(values, ["d1"] * 5 + ["d2"], ["a", "a", "a", "b", "c", "a"], ["central"] * 4 + ["peripheral", "central"])
        self.assertEqual(len(rows), 3)
        central = next(row for row in rows if row["donor_id"] == "d1" and row["retinal_region"] == "central")
        self.assertEqual(central["mean"], [5.])
        self.assertEqual(central["n_sections"], 2)
        self.assertEqual(central["n_cells"], 4)
        self.assertEqual(central["section_ids"], ["a", "b"])
        json.dumps(rows, allow_nan=False)

    def test_scalar_composite_keys_and_invalid_values(self):
        rows = pool_sections_to_donors([2., 4.], ["d", "d"], [("sample", "a"), ("sample", "b")], ["region", "region"])
        self.assertEqual(rows[0]["mean"], [3.])
        json.dumps(rows)
        with self.assertRaises(ValueError):
            pool_sections_to_donors([[float("nan")]], ["d"], ["s"], ["r"])
        with self.assertRaises(ValueError):
            pool_sections_to_donors([[1.]], ["d"], [None], ["r"])
        self.assertEqual(pool_sections_to_donors(np.empty((0, 2)), [], [], []), [])


if __name__ == "__main__":
    unittest.main()
