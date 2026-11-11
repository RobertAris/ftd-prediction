"""Synthetic filesystem checks: no study inputs, packages or remote operations."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pmacs_discover_inputs.py"
SPEC = importlib.util.spec_from_file_location("pmacs_discovery", SCRIPT)
discovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(discovery)
SECRET = "PLANTED_PRIVATE_IDENTIFIER"


@unittest.skipUnless(discovery.supported(), "requires POSIX no-follow directory handles")
class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "selected_stage"
        self.root.mkdir()
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, extra=(), roots=None, output=None):
        self.sequence += 1
        output = output or self.base / f"private_report_{self.sequence}"
        argv = [item for root in (roots or [self.root]) for item in ("--root", str(root))]
        argv += ["--output", str(output), *map(str, extra)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = discovery.main(argv)
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn(SECRET, stdout.getvalue())
        self.assertNotIn(str(self.base), stdout.getvalue())
        self.assertNotIn("Traceback", stdout.getvalue())
        return code, json.loads(stdout.getvalue()), output

    def report(self, output):
        return json.loads((output / "report.json").read_text())

    def test_metadata_only_never_opens_assays_tables_or_unselected_scripts(self):
        names = ["donor_mapping.csv", "section_manifest.tsv", "sample_sheet.xlsx", "metadata.rds",
                 "processed.H5AD", "processed.h5seurat", "counts.mtx.gz", "preprocess.R", "source.py"]
        for name in names:
            (self.root / name).write_text(SECRET)
        actual_open = os.open
        def guarded_open(path, flags, *args, **kwargs):
            # Only directory handles or exclusive report writes may be opened.
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertTrue(flags & os.O_DIRECTORY or flags & os.O_WRONLY)
            return actual_open(path, flags, *args, **kwargs)
        with patch.object(discovery, "supported", return_value=True), \
                patch.object(discovery.os, "open", side_effect=guarded_open), \
                patch.object(discovery.os, "read", side_effect=AssertionError("content read")), \
                patch("builtins.open", side_effect=AssertionError("content read")), \
                patch.object(Path, "open", side_effect=AssertionError("content read")):
            code, public, output = self.invoke()
        report = self.report(output)
        self.assertEqual(code, 0)
        self.assertEqual(len(report["candidates"]), len(names))
        self.assertEqual(public["counts"]["source_bytes_read"], 0)
        self.assertNotIn(SECRET, (output / "report.json").read_text())
        self.assertTrue(all(not c["verified"] for c in report["candidates"]))

    def test_pruning_and_atomic_zarr_are_never_entered(self):
        pruned = list(discovery.PRUNE_NAMES) + ["FASTQ_inputs", "aligned_BAM", "cellranger_run",
                                               "cellbender_run", "scrublet_run", "intermediates",
                                               "my_cache", "models", "checkpoints"]
        for name in pruned:
            directory = self.root / name
            directory.mkdir()
            (directory / f"donor_{SECRET}.csv").touch()
        store = self.root / "processed_model.ZARR"
        store.mkdir()
        (store / f"section_{SECRET}.csv").touch()
        code, public, output = self.invoke()
        report = self.report(output)
        self.assertEqual(code, 0)
        self.assertEqual(public["counts"]["directories_scanned"], 1)
        self.assertEqual(public["counts"]["directories_pruned"], len(pruned))
        self.assertEqual([c["kind"] for c in report["candidates"]], ["atomic_zarr_candidate"])
        self.assertNotIn(SECRET, json.dumps(report))
        code, _, output = self.invoke(roots=[store])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.report(output)["candidates"]), 1)

    def test_nested_symlinks_skipped_and_selected_root_alias_recorded(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / f"donor_{SECRET}.csv").touch()
        (self.root / "nested").symlink_to(outside, target_is_directory=True)
        (self.root / "donor_link.csv").symlink_to(outside / f"donor_{SECRET}.csv")
        (self.root / "broken").symlink_to(outside / "missing")
        (self.root / "inside.h5ad").touch()
        alias = self.base / "stage_alias"
        alias.symlink_to(self.root, target_is_directory=True)
        code, public, output = self.invoke(roots=[alias])
        self.assertEqual(code, 0)
        self.assertEqual(public["counts"]["symlinks_skipped"], 3)
        report = self.report(output)
        self.assertEqual(report["roots"][0], {"selected_path": str(alias), "canonical_path": str(self.root)})
        self.assertEqual(len(report["candidates"]), 1)

    def test_queued_directory_symlink_replacement_cannot_escape(self):
        queued = self.root / "queued"
        queued.mkdir()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / f"donor_{SECRET}.csv").touch()
        actual = discovery.open_directory
        def raced(root_fd, parts):
            if parts == ("queued",):
                queued.rmdir()
                queued.symlink_to(outside, target_is_directory=True)
            return actual(root_fd, parts)
        with patch.object(discovery, "open_directory", side_effect=raced):
            code, public, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("directory_error", public["reason_codes"])
        self.assertEqual(self.report(output)["candidates"], [])

    def test_depth_zero_does_not_visit_child(self):
        (self.root / "root_mapping.csv").touch()
        child = self.root / "deeper"
        child.mkdir()
        (child / "donor_mapping.csv").touch()
        code, public, output = self.invoke(["--max-depth", "0"])
        self.assertEqual(code, 2)
        self.assertEqual(public["truncation_reasons"], ["max_depth"])
        self.assertEqual(public["counts"]["directories_skipped_depth"], 1)
        self.assertEqual(len(self.report(output)["candidates"]), 1)

    def test_entry_limit_is_global_across_roots(self):
        other = self.base / "other_stage"
        other.mkdir()
        for root in (self.root, other):
            for number in range(4):
                (root / f"uninteresting_{number}.bin").touch()
        code, public, _ = self.invoke(["--max-entries", "5"], roots=[self.root, other])
        self.assertEqual(code, 2)
        self.assertEqual(public["counts"]["entries_examined"], 5)
        self.assertEqual(public["counts"]["roots_scanned"], 2)
        self.assertIn("max_entries", public["truncation_reasons"])

    def test_candidate_limit_stops_traversal(self):
        for number in range(5):
            (self.root / f"donor_mapping_{number}.csv").touch()
        code, public, output = self.invoke(["--max-candidates", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(public["counts"]["entries_examined"], 2)
        self.assertEqual(len(self.report(output)["candidates"]), 2)
        self.assertIn("max_candidates", public["truncation_reasons"])

    def test_private_modes_umask_restored_and_no_overwrite(self):
        previous = os.umask(0)
        try:
            code, _, output = self.invoke()
            observed = os.umask(0)
            self.assertEqual(observed, 0)
        finally:
            os.umask(previous)
        self.assertEqual(code, 0)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        for name in ("report.json", "snippets.txt"):
            self.assertEqual(stat.S_IMODE((output / name).stat().st_mode), 0o600)
        original = (output / "report.json").read_bytes()
        code, _, _ = self.invoke(output=output)
        self.assertEqual(code, 1)
        self.assertEqual((output / "report.json").read_bytes(), original)

    def test_output_file_and_symlink_refused_without_following(self):
        target = self.base / "existing_file"
        target.write_text(SECRET)
        link = self.base / "output_link"
        link.symlink_to(target)
        broken = self.base / "broken_output"
        broken.symlink_to(self.base / "missing")
        for output in (target, link, broken):
            code, _, _ = self.invoke(output=output)
            self.assertEqual(code, 1)
        self.assertEqual(target.read_text(), SECRET)

    def test_output_canonical_boundary_checked_before_scan(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with patch.object(discovery.os, "scandir", side_effect=AssertionError("must not scan")), \
                patch.object(discovery, "supported", return_value=True):
            for output in (self.root / "private", alias / "private"):
                code, public, _ = self.invoke(output=output)
                self.assertEqual(code, 1)
                self.assertEqual(public["reason_codes"], ["output_inside_root"])
                self.assertFalse(output.exists())

    def test_unsafe_roots_placeholders_and_invalid_arguments_are_sanitized(self):
        for root in ("/", "/project", "/project/hipaa_ycheng11lab", "/project/<source>/data", "relative/stage",
                     f"/project/REPLACE_ME/{SECRET}", str(self.base / SECRET / "missing")):
            code, _, _ = self.invoke(roots=[root])
            self.assertEqual(code, 1)
        for extra in (["--unexpected", SECRET], ["--max-depth", SECRET], ["--max-depth", "-1"],
                      ["--max-entries", "0"], ["--max-entries", "100001"], ["--max-candidates", "2001"],
                      ["--max-source-bytes", "1048577"], ["--max-source-lines", "401"],
                      ["--max-line-length", "1001"], ["--max-depth", "17"]):
            code, _, _ = self.invoke(extra)
            self.assertEqual(code, 1)
        self.assertEqual(self.invoke(roots=[self.root] * 17)[0], 1)
        self.assertEqual(self.invoke(["--source-file", str(self.root / "source.py")] * 17)[0], 1)

    def test_exact_source_selection_and_line_numbers_are_private(self):
        source = self.root / f"preprocess_{SECRET}.R"
        source.write_text(f"# irrelevant\nx <- readRDS('{SECRET}')\nx@assays$RNA@layers\nNormalizeData(x)\n")
        unselected = self.root / "not_selected.py"
        unselected.write_text("save('DO_NOT_EXTRACT')\n")
        code, public, output = self.invoke(["--source-file", str(source)])
        self.assertEqual(code, 0)
        record = self.report(output)["sources"][0]
        self.assertEqual([s["line"] for s in record["snippets"]], [2, 3, 4])
        self.assertEqual(public["counts"]["source_lines_emitted"], 3)
        self.assertIn(SECRET, (output / "snippets.txt").read_text())
        self.assertNotIn("DO_NOT_EXTRACT", json.dumps(self.report(output)))

    def test_source_bytes_lines_and_line_lengths_are_bounded(self):
        source = self.root / "preprocess.py"
        source.write_text("read('safe')\n" + "x" * 500)
        code, public, output = self.invoke(["--source-file", str(source), "--max-source-bytes", "20"])
        self.assertEqual(code, 2)
        self.assertEqual(public["counts"]["source_bytes_read"], 20)
        record = self.report(output)["sources"][0]
        self.assertIn("max_source_bytes", record["limitations"])
        self.assertIn("partial_last_line_omitted", record["limitations"])
        self.assertEqual(len(record["snippets"]), 1)
        source.write_text("donor = '" + SECRET * 10 + "'\nsection = 1\nread('safe')\n")
        other = self.root / "second.R"
        other.write_text("save(x)\n")
        code, public, output = self.invoke(["--source-file", str(source), "--source-file", str(other),
                                            "--max-source-lines", "2", "--max-line-length", "12"])
        self.assertEqual(code, 2)
        report = self.report(output)
        self.assertEqual(public["counts"]["source_lines_emitted"], 2)
        self.assertEqual(report["sources"][1]["bytes_read"], 0)
        self.assertTrue(all(len(s["text"]) <= 12 for r in report["sources"] for s in r["snippets"]))
        self.assertEqual(set(public["truncation_reasons"]), {"max_source_lines", "max_line_length"})

    def test_source_boundary_extensions_symlinks_and_nonregular_refused(self):
        outside = self.base / "outside.py"
        outside.write_text(f"read('{SECRET}')")
        table = self.root / "mapping.csv"
        table.write_text(SECRET)
        link = self.root / "linked.py"
        link.symlink_to(outside)
        directory = self.root / "directory.py"
        directory.mkdir()
        sublink = self.root / "linked_directory"
        sublink.symlink_to(self.base, target_is_directory=True)
        fifo = self.root / "pipe.py"
        os.mkfifo(fifo)
        store = self.root / "atomic.zarr"
        store.mkdir()
        (store / "source.py").write_text(SECRET)
        selected = [outside, table, link, directory, sublink / "outside.py", fifo,
                    self.root / ".." / "outside.py", store / "source.py"]
        with patch.object(discovery.os, "read", side_effect=AssertionError("forbidden read")):
            for source in selected:
                code, public, output = self.invoke(["--source-file", str(source)])
                self.assertEqual(code, 1)
                self.assertEqual(public["counts"]["source_bytes_read"], 0)
                self.assertEqual(self.report(output)["sources"][0]["status"], "error")

    def test_selected_source_through_explicit_root_alias_is_allowed(self):
        source = self.root / "source.py"
        source.write_text("log1p(x)\n")
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        code, _, output = self.invoke(["--source-file", str(alias / source.name)], roots=[alias])
        self.assertEqual(code, 0)
        self.assertEqual(self.report(output)["sources"][0]["path"], str(source))

    def test_exception_details_stay_private(self):
        with patch.object(discovery.os, "scandir", side_effect=PermissionError(SECRET)), \
                patch.object(discovery, "supported", return_value=True):
            code, public, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(public["reason_codes"], ["directory_error"])
        self.assertIn(SECRET, self.report(output)["private_errors"][0]["detail"])

    def test_source_replaced_by_symlink_before_open_is_not_read(self):
        source = self.root / "source.py"
        source.write_text("read('safe')\n")
        outside = self.base / "outside.py"
        outside.write_text(SECRET)
        actual = os.open
        def raced(path, flags, *args, **kwargs):
            if path == source.name and not flags & os.O_WRONLY:
                source.unlink()
                source.symlink_to(outside)
            return actual(path, flags, *args, **kwargs)
        with patch.object(discovery.os, "open", side_effect=raced), \
                patch.object(discovery, "supported", return_value=True), \
                patch.object(discovery.os, "read", side_effect=AssertionError("forbidden read")):
            code, public, output = self.invoke(["--source-file", str(source)])
        self.assertEqual(code, 1)
        self.assertEqual(public["counts"]["source_bytes_read"], 0)
        self.assertEqual(self.report(output)["sources"][0]["status"], "error")

    def test_write_failure_is_sanitized_and_saved_privately_when_possible(self):
        actual = discovery.private_write
        def failed(output_fd, name, text):
            if name == "report.json":
                raise OSError(SECRET)
            return actual(output_fd, name, text)
        with patch.object(discovery, "private_write", side_effect=failed):
            code, public, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("report_write_error", public["reason_codes"])
        failure = json.loads((output / "failure.json").read_text())
        self.assertIn(SECRET, failure["private_errors"][0]["detail"])
        self.assertEqual(stat.S_IMODE((output / "failure.json").stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
