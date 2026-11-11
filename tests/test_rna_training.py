"""Invented-data checks; no test in this file launches neural training."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ftd_prediction.rna_training import (
    _agreement, _check_split, _config, _diagnostics, _fit_centroids, _fit_pca,
    _log_normalize, _pca_mse, _predict_centroids, _prepare, _select_features, run_rna,
)


ARRAYS = all(importlib.util.find_spec(n) for n in ("anndata", "scipy", "sklearn"))
NEURAL = ARRAYS and importlib.util.find_spec("scvi") is not None


def config(**overrides):
    cfg = dict(train_donors=["invented_train"], test_donors=["invented_test"],
                max_epochs=2, max_genes=6, n_hidden=8, n_latent=2, n_layers=1,
                batch_size=4, accelerator="cpu", max_minutes=1, num_threads=1)
    return {**cfg, **overrides}


def fixture():
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(np.random.default_rng(11).poisson(3, (16, 10)).astype(np.float32))
    data = ad.AnnData(X=matrix.copy(), obs=pd.DataFrame({
        "donor_id": ["invented_train"] * 8 + ["invented_test"] * 8,
        "cell_type": (["A"] * 4 + ["B"] * 4) * 2,
        "diagnosis": ["ignored"] * 16,
    }, index=[f"invented_cell_{i}" for i in range(16)]),
        var=pd.DataFrame(index=[f"gene_{i:02d}" for i in range(10)]))
    data.layers["counts"] = matrix.copy()
    return data


class RNAConfigurationTests(unittest.TestCase):
    def test_split_is_explicit_disjoint_and_not_inferred(self):
        for cfg in ({}, {"train_donors": ["a"]}, {"test_donors": []},
                    {"train_donors": [], "test_donors": []},
                    {"train_donors": ["a"], "test_donors": ["a"]},
                    {"train_donors": ["a", "a"], "test_donors": []},
                    {"train_donors": "a", "test_donors": []}):
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                _config(cfg)

    def test_missing_and_malformed_split_identifiers_rejected(self):
        for donor in (None, "", " ", "unknown", "unresolved", "restricted", "not_applicable", " x", "x\n"):
            with self.subTest(value=donor), self.assertRaises(ValueError):
                _config(dict(train_donors=[donor], test_donors=[]))

    def test_numeric_bounds_and_unrecognized_settings(self):
        valid = config()
        invalid = {"seed": [-1, True, 1.5, 2**32], "max_epochs": [0, 101, 1.5, True],
                   "max_minutes": [0, 61, float("nan"), float("inf"), True],
                   "max_genes": [1, 2001], "n_hidden": [0, 4097], "n_latent": [0, 513],
                   "n_layers": [0, 17], "batch_size": [1, 65537], "num_threads": [0, 129],
                   "learning_rate": [0, -1, float("nan"), 2], "weight_decay": [0, -1, float("inf")],
                   "kl_warmup_epochs": [0, 101], "accelerator": ["auto", "mps", None],
                   "batch_key": ["donor_id"], "labels_key": ["diagnosis"], "ckpt_path": ["unapproved"]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    _config({**valid, key: value})

    def test_defaults_and_caller_config_are_preserved(self):
        cfg = dict(train_donors=["a"], test_donors=[])
        before = copy.deepcopy(cfg)
        result = _config(cfg)
        self.assertEqual(cfg, before)
        self.assertEqual(result["seed"], 11)
        self.assertEqual(result["max_epochs"], 100)
        self.assertEqual(result["max_minutes"], 60)
        result["train_donors"].append("different")
        self.assertEqual(cfg, before)

    def test_actual_split_verifier_rejects_missing_duplicate_or_holdout_indices(self):
        _check_split(np.arange(4), [], [], 4)
        for train, val, test in ((None, [], []), ([0, 1, 2], [], []), ([0, 1, 2, 2], [], []),
                                 ([0, 1, 2, 3], [0], []), ([0, 1, 2, 3], [], [0])):
            with self.subTest(train=train, val=val, test=test), self.assertRaises(RuntimeError):
                _check_split(train, val, test, 4)

    def test_embedding_collapse_and_nonfinite_checks(self):
        collapsed = _diagnostics(np.ones((8, 3)))
        self.assertEqual(collapsed, dict(mean_latent_variance=0.0, rank=0, effective_rank=0.0))
        with self.assertRaises(RuntimeError):
            _diagnostics(np.array([[1.0, np.nan], [2.0, 1.0]]))


@unittest.skipUnless(ARRAYS, "Optional AnnData/SciPy/sklearn environment required")
class RNADataAndBaselineTests(unittest.TestCase):
    def test_holdout_counts_labels_and_diagnosis_cannot_change_features_or_pca(self):
        a = fixture()
        b = a.copy()
        replacement = b.layers["counts"].toarray()
        replacement[8:] = 1
        replacement[8:, 9] = 1_000_000
        from scipy.sparse import csr_matrix
        b.layers["counts"] = csr_matrix(replacement)
        b.obs.loc[b.obs.index[8:], "cell_type"] = "holdout_only_class"
        b.obs["diagnosis"] = "unrelated"
        first, second = _prepare(a, config()), _prepare(b, config())
        np.testing.assert_array_equal(first["selected"], second["selected"])
        np.testing.assert_array_equal(first["variance"], second["variance"])
        self.assertEqual((first["train"].X != second["train"].X).nnz, 0)
        self.assertEqual((first["train_log"] != second["train_log"]).nnz, 0)
        pca_a = _fit_pca(first["train_log"], 2, 11)
        pca_b = _fit_pca(second["train_log"], 2, 11)
        np.testing.assert_array_equal(pca_a.mean_, pca_b.mean_)
        np.testing.assert_array_equal(pca_a.components_, pca_b.components_)
        for prepared in (first, second):
            rows = prepared["train_rows"]
            categories, centers = _fit_centroids(prepared["train_log"],
                [prepared["labels"][i] for i in rows], prepared["donors"][rows])
            self.assertEqual(categories, ["A", "B"])
            if prepared is first:
                original = centers
            else:
                np.testing.assert_array_equal(original, centers)

    def test_sparse_preparation_and_pca_never_densify_expression(self):
        from scipy.sparse import csr_matrix
        with patch.object(csr_matrix, "toarray", side_effect=AssertionError("Dense expression forbidden")):
            prepared = _prepare(fixture(), config())
            pca = _fit_pca(prepared["train_log"], 2, 11)
            scores = pca.transform(prepared["train_log"])
            self.assertEqual(scores.shape, (8, 2))
            self.assertTrue(np.isfinite(scores).all())

    def test_pca_is_centered_and_sparse_reconstruction_mse_matches_direct_calculation(self):
        prepared = _prepare(fixture(), config())
        matrix = prepared["train_log"]
        pca = _fit_pca(matrix, 2, 11)
        scores = pca.transform(matrix)
        np.testing.assert_allclose(scores.mean(axis=0), 0, atol=1e-12)
        dense = matrix.toarray()
        expected = np.mean(np.square(dense - (scores @ pca.components_ + pca.mean_)))
        self.assertAlmostEqual(_pca_mse(matrix, scores, pca.mean_), expected, places=12)
        # The same fixed basis also gives the correct heldout reconstruction.
        query = _log_normalize(prepared["counts"][prepared["test_rows"]])[:, prepared["selected"]]
        query_scores = pca.transform(query)
        expected = np.mean(np.square(query.toarray() - (query_scores @ pca.components_ + pca.mean_)))
        self.assertAlmostEqual(_pca_mse(query, query_scores, pca.mean_), expected, places=12)

    def test_equal_variance_ties_resolve_by_gene_name(self):
        from scipy.sparse import csr_matrix
        selected, _ = _select_features(csr_matrix([[1, 1, 0], [2, 2, 0]]), ["z", "a", "m"], 2)
        self.assertEqual(selected.tolist(), [1, 0])

    def test_query_normalization_does_not_depend_on_other_query_cells(self):
        from scipy.sparse import csr_matrix
        first = _log_normalize(csr_matrix([[1, 2, 3]]))
        many = _log_normalize(csr_matrix([[1, 2, 3], [100000, 1, 0]]))
        self.assertEqual((first != many[:1]).nnz, 0)

    def test_preparation_preserves_all_caller_matrices_and_metadata(self):
        import pandas as pd
        data = fixture()
        data.obsm["private_extra"] = np.ones((16, 2))
        data.uns["private_extra"] = {"key": [1, 2]}
        original = data.copy()
        prepared = _prepare(data, config())
        self.assertEqual((data.X != original.X).nnz, 0)
        self.assertEqual((data.layers["counts"] != original.layers["counts"]).nnz, 0)
        pd.testing.assert_frame_equal(data.obs, original.obs)
        pd.testing.assert_frame_equal(data.var, original.var)
        np.testing.assert_array_equal(data.obsm["private_extra"], original.obsm["private_extra"])
        self.assertEqual(data.uns, original.uns)
        self.assertEqual(list(prepared["train"].obs.columns), [])
        self.assertEqual(prepared["train"].n_obs, 8)
        self.assertFalse(set(prepared["train"].obs_names) & set(data.obs_names))
        prepared["train"].layers["counts"].data[:] = 0
        self.assertEqual((data.layers["counts"] != original.layers["counts"]).nnz, 0)

    def test_counts_layer_not_X_defines_model_input(self):
        data = fixture()
        before = _prepare(data, config())
        data.X.data[:] = -100
        after = _prepare(data, config())
        np.testing.assert_array_equal(before["selected"], after["selected"])
        self.assertEqual((before["train"].X != after["train"].X).nnz, 0)

    def test_fractional_negative_nonfinite_and_oversized_counts_fail(self):
        for value in (0.5, -1, np.nan, np.inf, 2**25):
            data = fixture()
            data.layers["counts"].data[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                _prepare(data, config())

    def test_invalid_sparse_duplicates_are_not_hidden_by_summation(self):
        from scipy.sparse import csr_matrix
        data = fixture()
        counts = data.layers["counts"]
        # Add a negative duplicate to a positive entry without canonicalizing.
        values = np.insert(counts.data, 1, -1)
        indices = np.insert(counts.indices, 1, counts.indices[0])
        indptr = counts.indptr.copy()
        indptr[1:] += 1
        data.layers["counts"] = csr_matrix((values, indices, indptr), shape=counts.shape)
        with self.assertRaises(ValueError):
            _prepare(data, config())

    def test_zero_libraries_and_insufficient_training_cells_fail(self):
        data = fixture()
        data.layers["counts"].data[:data.layers["counts"].indptr[1]] = 0
        with self.assertRaises(ValueError):
            _prepare(data, config())
        with self.assertRaises(ValueError):
            _prepare(fixture()[[0, 1, 8, 9]].copy(), config())

    def test_boolean_counts_and_empty_selected_libraries_fail(self):
        from scipy.sparse import csr_matrix
        data = fixture()
        data.layers["counts"] = data.layers["counts"].astype(bool)
        with self.assertRaises(ValueError):
            _prepare(data, config())
        data = fixture()
        counts = data.layers["counts"].toarray()
        chosen = _prepare(data, config())["selected"]
        counts[8] = 0
        counts[8, next(i for i in range(10) if i not in chosen)] = 1
        data.layers["counts"] = csr_matrix(counts)
        with self.assertRaises(ValueError):
            _prepare(data, config())

    def test_unmapped_uncovered_and_unused_donors_fail_without_exposing_values(self):
        for value in (None, "unknown", "sensitive_unassigned_donor"):
            data = fixture()
            data.obs.iloc[0, data.obs.columns.get_loc("donor_id")] = value
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                _prepare(data, config())
            self.assertNotIn("sensitive_unassigned_donor", str(caught.exception))
        cfg = config()
        cfg["train_donors"].append("unused")
        with self.assertRaises(ValueError):
            _prepare(fixture(), cfg)

    def test_duplicate_names_missing_counts_and_missing_donor_column_fail(self):
        for change in (lambda a: setattr(a, "obs_names", ["same"] * 16),
                       lambda a: setattr(a, "var_names", ["same"] * 10),
                       lambda a: a.layers.pop("counts"), lambda a: a.obs.pop("donor_id")):
            data = fixture()
            change(data)
            with self.assertRaises(ValueError):
                _prepare(data, config())

    def test_singleton_minibatch_is_rejected_before_fitting(self):
        cfg = config()
        cfg["batch_size"] = 7
        with self.assertRaises(ValueError):
            _prepare(fixture(), cfg)

    def test_all_donor_descriptive_fit_is_explicit(self):
        cfg = config()
        cfg["train_donors"] = ["invented_train", "invented_test"]
        cfg["test_donors"] = []
        prepared = _prepare(fixture(), cfg)
        self.assertEqual(prepared["train"].n_obs, 16)
        self.assertEqual(len(prepared["test_rows"]), 0)

    def test_missing_annotation_tokens_are_excluded_from_prototypes(self):
        data = fixture()
        data.obs["cell_type"] = [None, np.nan, "unknown", "restricted", "unresolved", "not_applicable", "", "A"] * 2
        prepared = _prepare(data, config())
        self.assertEqual(prepared["labels"][:8], [None] * 7 + ["A"])
        categories, _ = _fit_centroids(prepared["train_log"], prepared["labels"][:8], prepared["donors"][:8])
        self.assertEqual(categories, ["A"])

    def test_donor_balanced_centroids_and_inference(self):
        from scipy.sparse import csr_matrix
        matrix = csr_matrix([[10, 0], [10, 0], [10, 0], [0, 10], [0, 20]])
        categories, centers = _fit_centroids(matrix, ["A", "A", "A", "A", "B"],
                                             np.asarray(["a", "a", "a", "b", "b"]))
        self.assertEqual(categories, ["A", "B"])
        np.testing.assert_allclose(centers, [[5, 5], [0, 20]])
        self.assertEqual(_predict_centroids(csr_matrix([[6, 4], [0, 19]]), categories, centers), ["A", "B"])
        empty_categories, empty = _fit_centroids(matrix, [None] * 5, np.array(["a"] * 5))
        self.assertEqual(empty_categories, [])
        self.assertEqual(empty.shape, (0, 2))

    def test_agreement_reports_unsupported_classes_and_missing_annotations(self):
        result = _agreement(["A", "B", None], ["A", "A", "A"], ["A"])
        self.assertEqual(result["n_labelled"], 2)
        self.assertEqual(result["unsupported_classes"], ["B"])
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertIsNone(_agreement([None], ["A"], ["A"])["macro_f1"])
        # A trained class predicted spuriously in a donor is also in macro-F1.
        self.assertAlmostEqual(_agreement(["A", "A"], ["A", "B"], ["A", "B"])["macro_f1"], 1 / 3)

    def test_existing_output_and_invalid_input_do_not_create_or_overwrite_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            cfg = config()
            cfg["test_donors"] = ["invented_train"]
            with self.assertRaises(ValueError):
                run_rna(fixture(), cfg, output)
            self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                run_rna(fixture(), config(), output)
            self.assertEqual(sentinel.read_text(), "preserve")


@unittest.skipUnless(NEURAL, "Optional scvi-tools environment required")
class RNAPinnedSplitterTests(unittest.TestCase):
    def test_actual_upstream_splitter_has_empty_validation_and_test(self):
        import scvi
        from scvi.dataloaders import DataSplitter

        train = _prepare(fixture(), config())["train"]
        scvi.model.SCVI.setup_anndata(train, layer="counts", batch_key=None, labels_key=None)
        # Constructing a model registers data; this test does not fit weights.
        model = scvi.model.SCVI(train, n_hidden=8, n_latent=2, n_layers=1)
        splitter = DataSplitter(model.adata_manager, train_size=1.0, validation_size=0.0,
                                batch_size=4, external_indexing=[np.arange(8), np.array([], dtype=int),
                                                               np.array([], dtype=int)])
        splitter.setup()
        _check_split(splitter.train_idx, splitter.val_idx, splitter.test_idx, 8)
        self.assertIsNone(splitter.val_dataloader())
        self.assertIsNone(splitter.test_dataloader())
        self.assertEqual(sum(len(batch["X"]) for batch in splitter.train_dataloader()), 8)


if __name__ == "__main__":
    unittest.main()
