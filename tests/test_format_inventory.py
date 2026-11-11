import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_formats.py"
_SPEC = importlib.util.spec_from_file_location("ftd_format_inventory", _SCRIPT)
inventory_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inventory_module)


@unittest.skipUnless(inventory_module._supported(), "requires POSIX no-follow directory handles")
class FormatInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.assay = self.root / "assay"
        self.assay.mkdir()
        # Environment metadata is tested separately; assay checks do not need
        # access to installed distribution metadata or import a scientific lib.
        self.versions = patch.object(inventory_module, "_versions", lambda result: None)
        self.versions.start()

    def tearDown(self):
        self.versions.stop()
        self.temp.cleanup()

    def test_case_insensitive_and_double_extensions(self):
        names = ["a.RDS", "b.rda", "c.H5AD", "d.h5Seurat", "e.H5", "f.MTX", "g.mTx.Gz",
                 "h.CSV", "i.CsV.GZ", "j.TSV", "k.tsv.gz", "l.PARQUET", "m.LOOM", "private_name.bin"]
        for name in names:
            (self.assay / name).touch()
        result = inventory_module.inventory(self.assay)
        self.assertTrue(result["complete"])
        self.assertEqual(result["regular_files"], len(names))
        for key in ("rds", "rda", "h5ad", "h5seurat", "h5", "mtx", "mtx.gz", "csv", "csv.gz", "tsv", "tsv.gz", "parquet", "loom", "unknown"):
            self.assertEqual(result["format_counts"][key], 1, key)
        self.assertNotIn("private_name", json.dumps(result))

    def test_assay_contents_are_never_opened(self):
        source = self.assay / "SENSITIVE_DONOR.rds"
        source.write_text("synthetic test content that the inventory must not read")
        actual_open = os.open
        flags_seen = []

        def directory_open(path, flags, *args, **kwargs):
            flags_seen.append(flags)
            self.assertTrue(flags & os.O_DIRECTORY)
            self.assertTrue(flags & os.O_NOFOLLOW)
            return actual_open(path, flags, *args, **kwargs)

        with patch("builtins.open", side_effect=AssertionError("file content read")), patch.object(Path, "open", side_effect=AssertionError("file content read")), patch.object(inventory_module.os, "open", side_effect=directory_open), patch.object(inventory_module, "_supported", return_value=True):
            result = inventory_module.inventory(self.assay)
        self.assertTrue(result["complete"])
        self.assertEqual(result["format_counts"]["rds"], 1)
        self.assertTrue(flags_seen)
        self.assertNotIn("SENSITIVE_DONOR", json.dumps(result))

    def test_child_symlinks_are_skipped_and_root_link_is_allowed(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "DO_NOT_COUNT.csv").touch()
        (self.assay / "inside.h5ad").touch()
        (self.assay / "directory_link").symlink_to(outside, target_is_directory=True)
        (self.assay / "file_link.h5ad").symlink_to(outside / "DO_NOT_COUNT.csv")
        (self.assay / "broken").symlink_to(outside / "missing")
        alias = self.root / "canonical_alias"
        alias.symlink_to(self.assay, target_is_directory=True)
        result = inventory_module.inventory(alias)
        self.assertTrue(result["complete"])
        self.assertEqual(result["symlinks_skipped"], 3)
        self.assertEqual(result["format_counts"]["csv"], 0)
        self.assertEqual(result["format_counts"]["h5ad"], 1)

    def test_directory_replaced_with_symlink_is_not_followed(self):
        child = self.assay / "child"
        child.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "DO_NOT_COUNT.csv").touch()
        actual = inventory_module._open_relative_directory

        def swap(root_fd, parts):
            if parts == ("child",):
                child.rename(self.root / "moved")
                child.symlink_to(outside, target_is_directory=True)
            return actual(root_fd, parts)

        with patch.object(inventory_module, "_open_relative_directory", side_effect=swap):
            result = inventory_module.inventory(self.assay)
        self.assertFalse(result["complete"])
        self.assertEqual(result["format_counts"]["csv"], 0)
        self.assertEqual(result["symlinks_skipped"], 1)
        self.assertGreater(result["error_count"], 0)

    def test_depth_cap_is_explicit_and_incomplete(self):
        (self.assay / "root.csv").touch()
        child = self.assay / "child"
        child.mkdir()
        (child / "child.h5ad").touch()
        capped = inventory_module.inventory(self.assay, max_depth=0)
        self.assertFalse(capped["complete"])
        self.assertTrue(capped["truncated"])
        self.assertEqual(capped["truncation_reasons"], ["max_depth"])
        self.assertEqual(capped["format_counts"]["csv"], 1)
        self.assertEqual(capped["format_counts"]["h5ad"], 0)
        complete = inventory_module.inventory(self.assay, max_depth=1)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["format_counts"]["h5ad"], 1)

    def test_hard_depth_ceiling_rejects_before_root_open_or_scan(self):
        for depth in (inventory_module.HARD_MAX_DEPTH + 1, 10**9):
            with self.subTest(depth=depth), patch.object(inventory_module.os, "open") as opener, patch.object(inventory_module.os, "scandir") as scanner:
                result = inventory_module.inventory(self.assay, max_depth=depth)
            opener.assert_not_called()
            scanner.assert_not_called()
            self.assertFalse(result["complete"])
            self.assertIn("invalid_limits", result["reason_codes"])
            self.assertEqual(result["limits"]["hard_max_depth"], 16)
        boundary = inventory_module.inventory(self.assay, max_depth=inventory_module.HARD_MAX_DEPTH)
        self.assertTrue(boundary["complete"])

    def test_global_entry_cap_and_exact_limit_completion(self):
        for index in range(4):
            (self.assay / f"file{index}.csv").touch()
        result = inventory_module.inventory(self.assay, max_entries=2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["entries_examined"], 2)
        self.assertEqual(result["format_counts"]["csv"], 2)
        self.assertIn("max_entries", result["truncation_reasons"])
        exact = inventory_module.inventory(self.assay, max_entries=4)
        self.assertTrue(exact["complete"])
        self.assertEqual(exact["entries_examined"], 4)

    def test_zarr_is_an_atomic_directory_container(self):
        store = self.assay / "counts.ZARR"
        store.mkdir()
        (store / "private_chunk.bin").touch()
        result = inventory_module.inventory(self.assay, max_depth=0)
        self.assertTrue(result["complete"])
        self.assertEqual(result["format_counts"]["zarr"], 1)
        self.assertEqual(result["format_counts"]["unknown"], 0)
        direct = inventory_module.inventory(store)
        self.assertTrue(direct["complete"])
        self.assertEqual(direct["format_counts"]["zarr"], 1)
        self.assertEqual(direct["entries_examined"], 0)

    def test_missing_root_never_falls_back_to_existing_parent(self):
        (self.assay / "DO_NOT_COUNT.csv").touch()
        missing = self.assay / "SENSITIVE_MISSING_ROOT"
        with patch.object(inventory_module, "_supported", return_value=True), patch.object(inventory_module.os, "scandir") as scanner:
            result = inventory_module.inventory(missing)
        scanner.assert_not_called()
        self.assertFalse(result["complete"])
        self.assertGreater(result["error_count"], 0)
        text = json.dumps(result)
        self.assertNotIn(str(self.assay), text)
        self.assertNotIn("SENSITIVE_MISSING_ROOT", text)
        self.assertEqual(result["format_counts"]["csv"], 0)

    def test_permission_failure_is_counted_without_message_or_path(self):
        with patch.object(inventory_module, "_supported", return_value=True), patch.object(inventory_module.os, "scandir", side_effect=PermissionError("SENSITIVE_USER /private/input")):
            result = inventory_module.inventory(self.assay)
        self.assertFalse(result["complete"])
        self.assertEqual(result["permission_error_count"], 1)
        self.assertEqual(result["error_count"], 1)
        self.assertNotIn("SENSITIVE_USER", json.dumps(result))
        self.assertNotIn("/private/input", json.dumps(result))

    def test_cli_exit_codes_and_sanitized_argument_failures(self):
        (self.assay / "one.csv").touch()
        (self.assay / "two.csv").touch()
        cases = [(["--root", str(self.assay)], 0),
                 (["--root", str(self.assay), "--max-entries", "1"], 2),
                 (["--root", str(self.assay / "SENSITIVE_MISSING")], 1),
                 (["--root", str(self.assay), "--max-entries", "100001"], 1),
                 (["--root", str(self.assay), "--max-depth", "-1"], 1),
                 (["--root", str(self.assay), "--max-depth", "17"], 1),
                 (["--root", str(self.assay), "--max-depth", "SENSITIVE_INVALID_VALUE"], 1),
                 (["--root", ""], 1),
                 ([], 1)]
        for arguments, expected in cases:
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(arguments=arguments), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = inventory_module.main(arguments)
            self.assertEqual(code, expected)
            payload = json.loads(stdout.getvalue())
            self.assertIn("installed_versions", payload)
            self.assertEqual(stderr.getvalue(), "")
            self.assertNotIn(str(self.assay), stdout.getvalue())
            self.assertNotIn("SENSITIVE_", stdout.getvalue())


class EnvironmentMetadataTests(unittest.TestCase):
    def test_versions_are_queried_through_distribution_metadata_only(self):
        before = set(sys.modules)
        result = inventory_module._payload(3, 10000)
        with patch.object(inventory_module.metadata, "version", return_value="1.2.3") as lookup:
            inventory_module._versions(result)
        self.assertEqual(lookup.call_count, len(inventory_module.PACKAGES))
        self.assertEqual(result["installed_versions"], {name: "1.2.3" for name in inventory_module.PACKAGES})
        self.assertFalse(any(name in set(sys.modules) - before for name in ("torch", "novae", "scanpy", "scvi")))

    def test_absent_package_is_not_an_error_and_lookup_failure_is_sanitized(self):
        result = inventory_module._payload(3, 10000)
        with patch.object(inventory_module.metadata, "version", side_effect=inventory_module.metadata.PackageNotFoundError()):
            inventory_module._versions(result)
        self.assertEqual(result["error_count"], 0)
        with patch.object(inventory_module.metadata, "version", side_effect=PermissionError("SENSITIVE_METADATA_PATH")):
            inventory_module._versions(result)
        self.assertEqual(result["version_lookup_error_count"], len(inventory_module.PACKAGES))
        self.assertEqual(result["permission_error_count"], len(inventory_module.PACKAGES))
        self.assertNotIn("SENSITIVE_METADATA_PATH", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
