"""Invented spatial inputs only. Integration tests require ``ailab compute``."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ("anndata", "scipy", "sklearn"))
NEURAL = AVAILABLE and importlib.util.find_spec("novae") is not None


def fixture():
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(11)
    grid = np.array([(x, y) for x in range(8) for y in range(8)], dtype=float) * 10
    xy = np.tile(grid, (3, 1)) + rng.normal(0, .03, (192, 2))
    rate = np.ones((192, 32))
    rate[xy[:, 0] < 40, :8] = 6
    rate[xy[:, 0] >= 40, 8:16] = 6
    counts = csr_matrix(rng.poisson(rate).astype("float32"))
    a = ad.AnnData(X=counts.copy(), obs=pd.DataFrame({
        "donor_id": ["invented_train"] * 128 + ["invented_test"] * 64,
        "section_id": ["invented_a"] * 64 + ["invented_b"] * 64 + ["invented_c"] * 64,
        "cell_type": ["left" if x < 40 else "right" for x in xy[:, 0]],
        "diagnosis": ["unused"] * 192,
    }, index=[f"invented_cell_{i}" for i in range(192)]),
        var=pd.DataFrame(index=[f"invented_gene_{i:02d}" for i in range(32)]))
    a.layers["counts"] = counts.copy()
    a.obsm["spatial"] = xy
    return a


def config():
    return dict(mode="synthetic_smoke", train_donors=["invented_train"],
                test_donors=["invented_test"], seed=11, max_epochs=2,
                batch_size=16, accelerator="cpu", num_workers=0,
                max_minutes=2, radius_um=18.0, domain_level=2)


@unittest.skipUnless(AVAILABLE, "Optional sparse/AnnData environment required")
class SpatialValidationTests(unittest.TestCase):
    """No neural fitting; these may run directly in the optional interpreter."""

    def reject(self, a=None, cfg=None):
        from ftd_prediction.spatial_training import run_spatial
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "bad"
            with self.assertRaises(ValueError):
                run_spatial(fixture() if a is None else a, config() if cfg is None else cfg, out)
            self.assertFalse(out.exists(), "Invalid inputs must fail before artifact creation")

    def test_donor_leakage_and_incomplete_or_implicit_splits(self):
        for changed in ({"test_donors": ["invented_train", "invented_test"]},
                        {"test_donors": []}, {"train_donors": []},
                        {"train_donors": ["invented_train", "invented_train"]},
                        {"train_donors": "invented_train"},
                        {"test_donors": ["invented_test", "absent"]}):
            with self.subTest(changed=changed):
                self.reject(cfg={**config(), **changed})
        cfg = config()
        del cfg["test_donors"]
        self.reject(cfg=cfg)

    def test_missing_and_conflicting_mappings(self):
        for column in ("donor_id", "section_id", "tissue_component"):
            a = fixture()
            if column == "tissue_component":
                a.obs[column] = "component"
            a.obs.loc[a.obs.index[0], column] = None
            self.reject(a)
        a = fixture()
        a.obs["section_id"] = "shared_across_donors"
        self.reject(a)

    def test_counts_and_coordinates_are_validated_without_echoing_ids(self):
        import numpy as np
        from ftd_prediction.spatial_training import _validate
        for invalid in (-1, .5, np.nan, np.inf):
            a = fixture()
            a.layers["counts"].data[0] = invalid
            with self.subTest(invalid=invalid):
                self.reject(a)
        a = fixture()
        del a.layers["counts"]
        self.reject(a)
        a = fixture()
        a.layers["counts"][0] = 0
        a.layers["counts"].eliminate_zeros()
        self.reject(a)
        a = fixture()
        a.obsm["spatial"][0, 0] = np.nan
        self.reject(a)
        a = fixture()
        a.obs.loc[a.obs.index[0], "donor_id"] = "private_identifier_do_not_echo"
        with self.assertRaises(ValueError) as caught:
            _validate(a, config())
        self.assertNotIn("private_identifier_do_not_echo", str(caught.exception))

    def test_duplicate_names_including_novae_case_collisions(self):
        for change in ("cells", "genes", "case"):
            a = fixture()
            if change == "cells":
                a.obs_names = ["duplicate"] * a.n_obs
            else:
                names = list(a.var_names)
                names[1] = names[0] if change == "genes" else names[0].upper()
                a.var_names = names
            self.reject(a)

    def test_bounds_unknown_settings_and_no_checkpoint_fallback(self):
        for key, value in (("max_epochs", 3), ("max_epochs", True), ("max_epochs", 1.5),
                           ("radius_um", 0), ("max_minutes", float("nan")),
                           ("learning_rate", float("inf")), ("accelerator", "auto"),
                           ("num_workers", -1), ("max_genes", 19), ("domain_level", 0), ("domain_level", 9),
                           ("resume", True), ("diagnosis_key", "diagnosis"),
                           ("checkpoint_path", "not_allowed_for_scratch")):
            with self.subTest(key=key, value=value):
                self.reject(cfg={**config(), key: value})
        for mode in ("frozen", "fine_tune"):
            self.reject(cfg={**config(), "mode": mode})
            self.reject(cfg={**config(), "mode": mode,
                             "checkpoint_path": "prism-oncology/novae-human-0"})

    def test_checkpoint_compatibility_is_explicit_and_expression_only(self):
        from ftd_prediction.novae_checkpoint import COMPATIBILITY
        from ftd_prediction.spatial_training import _validate
        cfg, _, _ = _validate(fixture(), config())
        self.assertEqual(cfg["checkpoint_compatibility"], "strict")
        for mode in (None, False, "auto", "permissive", "novae-human-0"):
            with self.subTest(mode=mode):
                self.reject(cfg={**config(), "checkpoint_compatibility": mode})
        self.reject(cfg={**config(), "checkpoint_compatibility": COMPATIBILITY})
        import numpy as np
        a = fixture()
        a.obsm["histo_embeddings"] = np.zeros((a.n_obs, 50))
        # This must fail before any artifact or stripped-down section is made.
        self.reject(a, {**config(), "mode": "frozen", "checkpoint_compatibility": COMPATIBILITY})

    def test_features_and_baseline_do_not_fit_holdout(self):
        import numpy as np
        from scipy.sparse import csr_matrix
        from ftd_prediction.spatial_training import (_validate, _normalize,
                                                     _select_features, _expression_baseline)
        a = fixture()
        _, counts, train = _validate(a, config())
        original = _normalize(counts)
        genes = list(a.var_names)
        chosen = _select_features(original[train], genes, 20)
        # Make a different gene dominate all test cells without touching train.
        new_counts = counts.tolil()
        new_counts[~train, 31] = 1000000
        changed = _normalize(new_counts.tocsr())
        self.assertEqual((original[train] != changed[train]).nnz, 0)
        np.testing.assert_array_equal(chosen, _select_features(changed[train], genes, 20))
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            first = _expression_baseline(original, chosen, train, a, out, 11)
            before = np.load(out / "expression_baseline.npz")["components"].copy()
            second = _expression_baseline(changed, chosen, train, a, out, 11)
            after = np.load(out / "expression_baseline.npz")["components"]
            np.testing.assert_allclose(before, after, atol=0, rtol=0)
            self.assertIsNotNone(first["annotation_agreement"]["test"])
            self.assertEqual(first["annotation_agreement"]["train"], second["annotation_agreement"]["train"])
        tied = csr_matrix(np.ones((10, 32)))
        tie_genes = list(reversed(genes))
        ids = _select_features(tied, tie_genes, 20)
        self.assertEqual([tie_genes[i] for i in ids], sorted(genes)[:20])

    def test_fresh_sparse_objects_strip_diagnosis_and_split_components(self):
        from scipy import sparse
        from ftd_prediction.spatial_training import _validate, _normalize, _make_sections
        a = fixture()
        a.obs["tissue_component"] = ["left" if x < 40 else "right" for x in a.obsm["spatial"][:, 0]]
        cfg, counts, train = _validate(a, config())
        sections = _make_sections(a, counts, _normalize(counts), list(range(32)), train)
        self.assertEqual(len(sections), 4)
        for section in sections:
            self.assertTrue(sparse.issparse(section.X))
            self.assertTrue(sparse.issparse(section.layers["counts"]))
            self.assertEqual(set(section.obs["donor_id"]), set(cfg["train_donors"]))
            self.assertEqual(section.obs["section_id"].nunique(), 1)
            self.assertEqual(section.obs["tissue_component"].nunique(), 1)
            self.assertNotIn("diagnosis", section.obs)
            self.assertFalse(section.obsp)

    def test_graph_audit_rejects_cross_boundary_and_binary_weights(self):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        from ftd_prediction.spatial_training import _audit_graphs
        a = ad.AnnData(X=csr_matrix(np.ones((4, 20))), obs=pd.DataFrame({
            "donor_id": ["d"] * 4, "section_id": ["s"] * 4,
            "tissue_component": ["c"] * 4}, index=[f"c{i}" for i in range(4)]))
        a.obsm["spatial"] = np.array([[0, 0], [3, 4], [10, 0], [13, 4]])
        a.obsp["spatial_distances"] = csr_matrix(([5., 5.], ([0, 1], [1, 0])), shape=(4, 4))
        self.assertEqual(_audit_graphs([a], 6)["directed_edges"], 2)
        for column in ("donor_id", "section_id", "tissue_component"):
            bad = a.copy()
            bad.obs.loc[bad.obs.index[1], column] = "other"
            with self.assertRaises(ValueError):
                _audit_graphs([bad], 6)
        bad = a.copy()
        bad.obsp["spatial_distances"].data[:] = 1
        with self.assertRaises(ValueError):
            _audit_graphs([bad], 6)
        with self.assertRaises(ValueError):
            _audit_graphs([a], 4)

    def test_degenerate_sections_and_zero_shot_are_rejected(self):
        import numpy as np
        from types import SimpleNamespace
        from ftd_prediction.spatial_training import _validate, _normalize, _make_sections, _infer
        a = fixture()
        a.obsm["spatial"][:, 1] = 0
        _, counts, mask = _validate(a, config())
        with self.assertRaises(ValueError):
            _make_sections(a, counts, _normalize(counts), np.arange(32), mask)
        with self.assertRaises(ValueError):
            _infer(SimpleNamespace(mode=SimpleNamespace(zero_shot=True)), [], config(), float("inf"))

    def test_empty_test_split_is_explicitly_descriptive(self):
        from ftd_prediction.spatial_training import _validate
        cfg = {**config(), "train_donors": ["invented_train", "invented_test"], "test_donors": []}
        actual, _, mask = _validate(fixture(), cfg)
        self.assertTrue(mask.all())
        self.assertEqual(actual["test_donors"], [])

    def test_invalid_neighborhoods_are_excluded_from_collapse_and_label_metrics(self):
        import numpy as np
        from ftd_prediction.spatial_training import _latent_metrics
        a = fixture()[:4].copy()
        a.obs["neighborhood_valid"] = [True, True, False, False]
        a.obs["cell_type"] = ["a", "b", "a", "b"]
        a.obs["novae_domains_2"] = ["D0", "D1", "D0", "D0"]
        a.obsm["novae_latent"] = np.array([[1., 0], [0, 1], [0, 0], [0, 0]], dtype="float32")
        metrics = _latent_metrics([a], 2)
        self.assertEqual(metrics["invalid_neighborhoods"], 2)
        self.assertEqual(metrics["valid_neighborhoods"], 2)
        self.assertEqual(metrics["domain_occupancy"], {"D0": 1, "D1": 1})
        self.assertEqual(metrics["latent_rank"], 1)
        self.assertEqual(metrics["annotation_agreement"]["adjusted_rand_index"], 1.0)
        a.obs["neighborhood_valid"] = False
        with self.assertRaises(ValueError):
            _latent_metrics([a], 2)

    def test_existing_output_and_expired_runtime_are_rejected(self):
        from ftd_prediction.spatial_training import run_spatial, _check_time
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                run_spatial(fixture(), config(), root)
        with self.assertRaises(TimeoutError):
            _check_time(0)

    def test_private_logging_captures_preexisting_console_handlers_and_restores(self):
        import io
        import logging
        from ftd_prediction.spatial_training import _private_runtime
        console = io.StringIO()
        logger = logging.getLogger("ftd_spatial_test_private")
        handler = logging.StreamHandler(console)
        old_handlers, old_propagate, old_level = logger.handlers, logger.propagate, logger.level
        logger.handlers, logger.propagate, logger.level = [handler], False, logging.INFO
        try:
            with tempfile.TemporaryDirectory() as root:
                with _private_runtime(Path(root)):
                    logger.info("invented_private_marker")
                self.assertIn("invented_private_marker", (Path(root) / "runtime.log").read_text())
                self.assertEqual(console.getvalue(), "")
                self.assertIs(handler.stream, console)
        finally:
            logger.handlers, logger.propagate, logger.level = old_handlers, old_propagate, old_level
            handler.close()


@unittest.skipUnless(NEURAL, "Optional Novae environment required")
class SpatialNeuralTests(unittest.TestCase):
    """Actual model updates, checkpointing, frozen fitting and offline mapping."""

    def test_smoke_then_frozen_are_train_only_and_preserve_every_input(self):
        import numpy as np
        import novae
        import torch
        from novae._constants import Nums
        from novae.data.convert import ZScoreTorch
        from ftd_prediction.spatial_training import run_spatial, _file_hash
        a = fixture()
        original = a.copy()
        old_settings = (novae.settings.auto_preprocessing, novae.settings.disable_multimodal,
                        novae.settings.scale_to_microns, Nums.LAZY_LOADING_SIZE_THRESHOLD)
        prototype_calls, fit_calls, cache_modes = [], [], []
        init_prototypes = novae.Novae.init_prototypes
        fit = novae.Novae.fit
        converter_init = ZScoreTorch.__init__

        def prototype_spy(model, adata, *args, **kwargs):
            prototype_calls.append(set().union(*(set(s.obs["donor_id"]) for s in adata)))
            return init_prototypes(model, adata, *args, **kwargs)

        def fit_spy(model, *args, **kwargs):
            sections = kwargs.get("adata", model.adatas)
            fit_calls.append(set().union(*(set(s.obs["donor_id"]) for s in sections)))
            return fit(model, *args, **kwargs)

        def converter_spy(converter, *args, **kwargs):
            converter_init(converter, *args, **kwargs)
            cache_modes.append(converter.tensors is None)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(novae.Novae, "init_prototypes", prototype_spy), \
                patch.object(novae.Novae, "fit", fit_spy), \
                patch.object(ZScoreTorch, "__init__", converter_spy):
            output = Path(root) / "smoke"
            result = run_spatial(a, config(), output)
            self.assertTrue(result["encoder_updated"])
            self.assertTrue(result["reload_verified"])
            self.assertEqual(result["training_steps"], 8)
            self.assertFalse(result["metrics"]["masks_checked"])
            self.assertTrue(result["metrics"]["test_section_adaptation"])
            self.assertGreater(result["metrics"]["train"]["latent_rank"], 0)
            self.assertIsNotNone(result["metrics"]["expression_baseline"]["annotation_agreement"])
            history = json.loads((output / "training_history.json").read_text())
            for phase in history:
                self.assertEqual(phase["completed_epochs"], 2)
                self.assertTrue(all(np.isfinite(phase["losses"])))
                checkpoint = torch.load(output / phase["checkpoint"], weights_only=False, map_location="cpu")
                self.assertTrue(checkpoint["optimizer_states"])
                self.assertGreater(checkpoint["global_step"], 0)
            source_hash = _file_hash(output / "model" / "model.safetensors")
            frozen = run_spatial(a, {**config(), "mode": "frozen",
                                     "checkpoint_path": str(output / "model")}, Path(root) / "frozen")
            self.assertFalse(frozen["encoder_updated"])
            self.assertEqual(frozen["training_steps"], 0)
            self.assertTrue(frozen["reload_verified"])
            self.assertEqual(source_hash, _file_hash(output / "model" / "model.safetensors"))
            # An incomplete/mismatched checkpoint must not initialize silent random weights.
            from safetensors.torch import load_file, save_file
            saved = load_file(output / "model" / "model.safetensors")
            parameter = next(key for key, value in saved.items()
                             if key.startswith("encoder.") and value.ndim == 2)
            for mismatch in ("missing", "unexpected", "shape"):
                with self.subTest(checkpoint_mismatch=mismatch):
                    broken = Path(root) / mismatch
                    broken.mkdir()
                    (broken / "config.json").write_bytes((output / "model" / "config.json").read_bytes())
                    tensors = dict(saved)
                    if mismatch == "missing":
                        tensors.pop(parameter)
                    elif mismatch == "unexpected":
                        tensors["unexpected_tensor"] = torch.zeros(1)
                    else:
                        tensors[parameter] = tensors[parameter][:-1].clone()
                    save_file(tensors, broken / "model.safetensors")
                    with self.assertRaises(RuntimeError):
                        run_spatial(a, {**config(), "mode": "frozen",
                                        "checkpoint_path": str(broken)}, Path(root) / (mismatch + "_run"))
            for res, out in ((result, output), (frozen, Path(root) / "frozen")):
                json.dumps(res, allow_nan=False)
                self.assertIsNone(res["checkpoint_compatibility"])
                self.assertIsNone(json.loads((out / "provenance.json").read_text())["checkpoint_compatibility"])
                for relative in res["artifacts"].values():
                    path = (out / relative).resolve()
                    self.assertTrue(path.is_relative_to(out.resolve()))
                    self.assertTrue(path.exists())
        self.assertEqual(len(prototype_calls), 2)  # fine_tune train + frozen train
        self.assertEqual(len(fit_calls), 2)  # scratch + fine_tune
        self.assertTrue(all(donors == {"invented_train"} for donors in prototype_calls + fit_calls))
        self.assertTrue(cache_modes and all(cache_modes))
        self.assertEqual(old_settings, (novae.settings.auto_preprocessing, novae.settings.disable_multimodal,
                                       novae.settings.scale_to_microns, Nums.LAZY_LOADING_SIZE_THRESHOLD))
        self.assertEqual((a.X != original.X).nnz, 0)
        self.assertEqual((a.layers["counts"] != original.layers["counts"]).nnz, 0)
        self.assertTrue(a.obs.equals(original.obs))
        self.assertTrue(a.var.equals(original.var))
        self.assertEqual(set(a.obsm), set(original.obsm))
        self.assertEqual(set(a.obsp), set(original.obsp))
        np.testing.assert_array_equal(a.obsm["spatial"], original.obsm["spatial"])
        self.assertEqual(a.uns, copy.deepcopy(original.uns))


if __name__ == "__main__":
    unittest.main()
