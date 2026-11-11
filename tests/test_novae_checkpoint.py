"""Pinned public weights and invented expression only; managed compute checks.

No acquisition occurs. FTD_PUBLIC_NOVAE_DIR opts into the actual public smoke;
once set, a missing directory/dependency is a failure, never a skip.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ftd_prediction import novae_checkpoint as checkpoint


class LoaderBoundaryTests(unittest.TestCase):
    def test_unknown_modes_fail_without_loading(self):
        for mode in (None, False, [], {}, "", "auto", "strict=False", "novae-human-0"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                checkpoint.load_expression_checkpoint("absent", mode)

    def test_default_is_strict_and_never_retries_on_failure(self):
        loader = Mock(side_effect=RuntimeError("missing active weight"))
        novae = SimpleNamespace(__version__="1.1.3", Novae=SimpleNamespace(from_pretrained=loader))
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {"novae": novae}):
            with self.assertRaisesRegex(RuntimeError, "missing active weight"):
                checkpoint.load_expression_checkpoint(root)
            loader.assert_called_once_with(Path(root), local_files_only=True,
                                           map_location="cpu", strict=True)

    def test_unsupported_runtime_version_and_constructor(self):
        with tempfile.TemporaryDirectory() as root:
            novae = SimpleNamespace(__version__="1.1.4")
            with patch.dict("sys.modules", {"novae": novae}), self.assertRaisesRegex(ValueError, "1.1.3"):
                checkpoint.load_expression_checkpoint(root, checkpoint.COMPATIBILITY)
        def unexpected_constructor(self, **kwargs):
            pass
        novae = SimpleNamespace(__version__="1.1.3",
                                Novae=SimpleNamespace(__init__=unexpected_constructor))
        with self.assertRaisesRegex(ValueError, "constructor"):
            checkpoint._validate_runtime(novae)

    def test_file_hash_and_size_changes_are_rejected(self):
        for name in checkpoint.PINNED_FILES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                path = Path(root) / name
                original = b"tiny public stand-in"
                identity = {"size_bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}
                path.write_bytes(original)
                self.assertEqual(checkpoint._read_pinned_file(path, identity), original)
                path.write_bytes(b"X" + original[1:])
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    checkpoint._read_pinned_file(path, identity)
                path.write_bytes(original + b"X")
                with self.assertRaisesRegex(ValueError, "size"):
                    checkpoint._read_pinned_file(path, identity)
                with self.assertRaises(ValueError):
                    checkpoint._read_pinned_file(path, checkpoint.PINNED_FILES[name])

    def test_histo_input_rejected_at_both_adapter_and_encoder_boundaries(self):
        image = SimpleNamespace(obsm={"histo_embeddings": object()})
        with self.assertRaisesRegex(ValueError, "histo_embeddings"):
            checkpoint.reject_image_input([SimpleNamespace(obsm={}), image])
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, "H&E"):
            checkpoint.load_expression_checkpoint(root, checkpoint.COMPATIBILITY, adata=image)
        data = SimpleNamespace(histo_embeddings=None)
        for args, kwargs in (((data,), {}), ((), {"data": data})):
            with self.assertRaisesRegex(ValueError, "H&E"):
                checkpoint._reject_encoder_images(None, args, kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch environment required")
class TinyStateContractTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.expected = {name: torch.ones(shape) for name, shape in checkpoint.INACTIVE_SHAPES.items()}
        self.original = {"encoder.active.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                         "cell_embedder.active.bias": torch.tensor([-0.0, 2.0])}
        self.expected.update({name: torch.zeros_like(value) for name, value in self.original.items()})

    def test_only_exact_inactive_tensors_are_deterministically_completed(self):
        import torch
        complete = checkpoint._complete_expression_state(self.expected, self.original)
        self.assertEqual(set(complete), set(self.expected))
        self.assertEqual(set(complete) - set(self.original), set(checkpoint.INACTIVE_SHAPES))
        for name, tensor in self.original.items():
            self.assertIs(complete[name], tensor, "Active tensor objects must pass through unchanged")
        for name in checkpoint.INACTIVE_SHAPES:
            self.assertEqual(torch.count_nonzero(complete[name]).item(), 0)
        again = checkpoint._complete_expression_state(self.expected, self.original)
        for name in complete:
            self.assertTrue(torch.equal(complete[name].view(torch.uint8), again[name].view(torch.uint8)))
        self.assertEqual(len(self.original), 2, "Completion must not mutate the checkpoint")

    def test_all_active_mismatch_cases_fail(self):
        import torch
        for case in ("missing", "unexpected", "shape", "dtype", "partial_fusion", "complete_fusion"):
            original = dict(self.original)
            if case == "missing":
                original.pop("encoder.active.weight")
            elif case == "unexpected":
                original["arbitrary.extra.weight"] = torch.zeros(1)
            elif case == "shape":
                original["encoder.active.weight"] = torch.zeros(4, 3)
            elif case == "dtype":
                original["encoder.active.weight"] = original["encoder.active.weight"].double()
            elif case == "partial_fusion":
                original["encoder.mlp_fusion.0.bias"] = torch.zeros(114)
            else:
                original.update({name: self.expected[name] for name in checkpoint.INACTIVE_SHAPES})
            with self.subTest(case=case), self.assertRaises(ValueError):
                checkpoint._complete_expression_state(self.expected, original)

    def test_changed_current_fusion_contract_fails(self):
        import torch
        for change in (torch.zeros(113), torch.zeros(114, dtype=torch.float64)):
            expected = {**self.expected, "encoder.mlp_fusion.0.bias": change}
            with self.assertRaisesRegex(ValueError, "inactive"):
                checkpoint._complete_expression_state(expected, self.original)

    def test_current_full_state_schema_is_pinned(self):
        import torch
        self.assertNotEqual(checkpoint._schema_hash(self.expected), checkpoint.STATE_SCHEMA_SHA256)
        with self.assertRaisesRegex(ValueError, "dtype"):
            checkpoint._schema_hash({"x": torch.ones(1, dtype=torch.float64)})


class PublicCheckpointSmoke(unittest.TestCase):
    def test_actual_pinned_expression_checkpoint(self):
        if "FTD_PUBLIC_NOVAE_DIR" not in os.environ:
            self.skipTest("Explicit FTD_PUBLIC_NOVAE_DIR required; no public acquisition")
        value = os.environ["FTD_PUBLIC_NOVAE_DIR"]
        self.assertTrue(value, "FTD_PUBLIC_NOVAE_DIR must not be empty")
        directory = Path(value)
        self.assertTrue(directory.is_dir(), "Explicit public checkpoint directory is missing")
        import novae
        import torch
        from safetensors import safe_open
        from torch_geometric.data import Data
        from ftd_prediction.spatial_training import _state_hash

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), tempfile.TemporaryDirectory() as root:
                # Exercise the real default strict failure before any compatibility.
                with self.assertRaises(RuntimeError) as rejected:
                    checkpoint.load_expression_checkpoint(directory)
                for name in checkpoint.INACTIVE_SHAPES:
                    self.assertIn(name, str(rejected.exception))
                gc.collect()
                checkpoint._validate_runtime(novae)
                from novae._constants import Keys
                self.assertEqual(Keys.HISTO_EMBEDDINGS, checkpoint.HISTO_EMBEDDINGS)
                model, receipt = checkpoint.load_expression_checkpoint(directory, checkpoint.COMPATIBILITY)
                self.assertEqual(receipt["original_files"], checkpoint.PINNED_FILES)
                self.assertEqual(receipt["revision"], checkpoint.REVISION)
                self.assertEqual(receipt["active_tensor_count"], 78)
                self.assertEqual({v["name"] for v in receipt["synthesized_inactive_tensors"]},
                                 set(checkpoint.INACTIVE_SHAPES))
                self.assertTrue(receipt["strict_load_state_dict"])
                self.assertTrue(receipt["active_tensor_bytes_unchanged"])
                self.assertFalse(model.training)
                self.assertTrue(model.mode.pretrained)
                self.assertFalse(model.mode.multimodal)
                self.assertFalse(model.cell_embedder.embedding.weight.requires_grad)
                state = model.state_dict()
                with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as public:
                    self.assertEqual(set(state) - set(public.keys()), set(checkpoint.INACTIVE_SHAPES))
                    for name in public.keys():
                        self.assertTrue(torch.equal(state[name].view(torch.uint8),
                                                    public.get_tensor(name).view(torch.uint8)), name)
                for name in checkpoint.INACTIVE_SHAPES:
                    self.assertEqual(torch.count_nonzero(state[name]).item(), 0)
                del state

                # Exactly eight invented cells, 32 public-vocabulary genes, two
                # four-node graphs. Never allocate cells x the full vocabulary.
                genes = model.cell_embedder.gene_names[:32]
                indices = model.cell_embedder.genes_to_indices(genes).reshape(1, 32)
                counts = (torch.arange(8 * 32).reshape(8, 32) % 7 + 1).float()
                expression = torch.log1p(10000 * counts / counts.sum(dim=1, keepdim=True))
                edges = torch.tensor([(i, j) for start in (0, 4) for i in range(start, start + 4)
                                      for j in range(start, start + 4) if i != j]).T.contiguous()

                def invented_batch():
                    return Data(x=expression.clone(), genes_indices=indices.clone(),
                                edge_index=edges, edge_attr=torch.ones(edges.shape[1], 1),
                                batch=torch.tensor([0] * 4 + [1] * 4))

                before = _state_hash(model)
                with torch.no_grad():
                    latents = model.encoder(model.cell_embedder(invented_batch()))
                self.assertEqual(tuple(latents.shape), (2, 64))
                self.assertTrue(torch.isfinite(latents).all())
                self.assertFalse(latents.requires_grad)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(before, _state_hash(model), "Frozen inference must not update weights")
                bad = invented_batch()
                bad.histo_embeddings = torch.zeros(2, 50)
                with self.assertRaisesRegex(ValueError, "H&E"):
                    model.encoder(bad)
                with self.assertRaisesRegex(ValueError, "H&E"):
                    model.encoder(data=bad)

                exported = Path(root) / "complete-expression-model"
                model.save_pretrained(exported)
                with safe_open(exported / "model.safetensors", framework="pt", device="cpu") as public:
                    self.assertEqual(len(public.keys()), 84)
                # Keep at most one complete public model resident at a time.
                del model
                gc.collect()
                reloaded, strict_receipt = checkpoint.load_expression_checkpoint(exported)
                self.assertIsNone(strict_receipt)
                self.assertEqual(before, _state_hash(reloaded), "Every exported weight must reload exactly")
                self.assertFalse(reloaded.cell_embedder.embedding.weight.requires_grad)
                with torch.no_grad():
                    reload_latents = reloaded.encoder(reloaded.cell_embedder(invented_batch()))
                self.assertTrue(torch.equal(latents, reload_latents))
                self.assertEqual(before, _state_hash(reloaded))
                print("PUBLIC_NOVAE_CHECK " + json.dumps({
                    "status": "passed", "study_data_used": False, "network_acquisition": False,
                    "device": "cpu", "num_threads": 1, "invented_cells": 8, "genes": 32,
                    "max_resident_models": 1, "strict_original_rejected": True,
                    "active_bytes_equal": True, "frozen_inference_weights_unchanged": True,
                    "strict_export_reload_weights_equal": True, "strict_reload_latents_equal": True,
                    "latent_shape": list(latents.shape), "compatibility_receipt": receipt,
                }, sort_keys=True))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
