import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ftd_prediction.cli import load_config, main, protect_output_path, run_pilot, run_preflight, validate_runtime_scope
from ftd_prediction.provenance import file_sha256, source_snapshot, write_json
from ftd_prediction.reporting import inspect_run, write_report
from ftd_prediction.synthetic import generate_synthetic_fixtures


class RuntimeScopeTests(unittest.TestCase):
    def test_explicit_data_kind_and_lsf_context(self):
        validate_runtime_scope({"data_kind": "synthetic"}, mode="synthetic", execution_context="local", environ={})
        for mode, context, environment in [
            ("synthetic", "pmacs", {"LSB_JOBID": "23"}),
            ("study", "local", {"LSB_JOBID": "23"}),
            ("study", "pmacs", {}),
            ("study", "pmacs", {"LSB_JOBID": "not_a_job"}),
        ]:
            with self.subTest(mode=mode, context=context, env=environment), self.assertRaises(ValueError):
                validate_runtime_scope({"data_kind": "study"}, mode=mode, execution_context=context, environ=environment)
        validate_runtime_scope({"data_kind": "study"}, mode="study", execution_context="pmacs", environ={"LSB_JOBID": "23"})

    def test_input_root_and_symlink_outputs_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            canonical.mkdir()
            link = root / "alias"
            link.symlink_to(canonical, target_is_directory=True)
            for candidate in (canonical, canonical / "new_run", link / "new_run"):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    protect_output_path(candidate, [canonical])
            allowed = protect_output_path(root / "runs" / "unique", [canonical])
            self.assertEqual(allowed, (root / "runs" / "unique").resolve())
            self.assertFalse(allowed.exists())

    def test_existing_output_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                protect_output_path(tmp, [])

    def test_broken_output_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "absent_target"
            link = root / "existing_link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                protect_output_path(link, [])
            self.assertFalse(target.exists())

    def test_config_rejects_invalid_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for supplied in ({"max_cells": 0}, {"radius_um": float("nan")}, {"seed": True}, {"unreviewed_option": True}):
                path.write_text(json.dumps(supplied))
                with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                    load_config(path)

    def test_config_requires_at_least_two_null_permutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"n_permutations": 1}))
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(json.dumps({"n_permutations": 2}))
            self.assertEqual(load_config(path)["n_permutations"], 2)

    def test_cli_failure_does_not_repeat_private_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"private_value":"SENSITIVE_EXAMPLE", broken')
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["render-lsf", "--config", str(path), "--output", str(Path(tmp) / "job.lsf")])
            self.assertEqual(code, 1)
            self.assertNotIn("SENSITIVE_EXAMPLE", stdout.getvalue() + stderr.getvalue())

    def test_known_cluster_validation_reason_is_actionable_without_input_echo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text("{}")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["render-lsf", "--config", str(config), "--output", str(root / "job.lsf")])
            result = json.loads(stderr.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(result["reason_code"], "cluster_config_invalid")
            self.assertIn("workspace", result["message"])
            self.assertNotIn(str(root), result["message"])

    def test_argument_error_does_not_echo_submitted_value(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["pilot", "--mode", "SENSITIVE_EXAMPLE"])
        self.assertEqual(code, 1)
        self.assertNotIn("SENSITIVE_EXAMPLE", stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue())["reason_code"], "invalid_command_arguments")


class ProvenanceAndFixtureTests(unittest.TestCase):
    def test_source_digest_covers_uncommitted_source_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            source = root / "src" / "model.py"
            source.write_text("value=1\n")
            before = source_snapshot(root)
            source.write_text("value=2\n")
            after = source_snapshot(root)
            self.assertNotEqual(before["source_sha256"], after["source_sha256"])
            self.assertEqual(after["source_files"][0]["sha256"], file_sha256(source))
            self.assertIsNone(after["git_revision"])

    def test_seeded_fixtures_are_reproducible_and_unpaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = generate_synthetic_fixtures(root / "first", seed=29)
            second = generate_synthetic_fixtures(root / "second", seed=29)
            manifests = [json.loads(path.read_text()) for path in first.values()]
            rna_donors = {row["donor_id"] for row in manifests[0]["records"]}
            spatial_donors = {row["donor_id"] for row in manifests[1]["records"]}
            self.assertTrue(rna_donors.isdisjoint(spatial_donors))
            for name in first:
                self.assertEqual(file_sha256(first[name]), file_sha256(second[name]))
                self.assertEqual(file_sha256(first[name].parent / "counts.csv"), file_sha256(second[name].parent / "counts.csv"))
            with self.assertRaises(FileExistsError):
                generate_synthetic_fixtures(root / "first", seed=29)


class ReportTests(unittest.TestCase):
    def _run(self, root: Path):
        run = root / "run"
        run.mkdir()
        summary = {"schema_version": 1, "data_kind": "synthetic", "counts": {"n_cells": 12, "n_genes": 6, "n_donors": 2, "n_sections": 0}}
        write_json(run / "summary.json", summary)
        record = {"status": "complete", "data_kind": "synthetic", "seed": 11, "code": {"source_sha256": "abc"},
                  "artifacts": [{"path": "summary.json", "sha256": file_sha256(run / "summary.json")}],
                  "started_at": "2026-11-10T00:00:00+00:00", "finished_at": "2026-11-10T00:00:01+00:00"}
        write_json(run / "run.json", record)
        return run

    def test_report_states_synthetic_scope_and_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root)
            self.assertEqual(inspect_run(run)["counts"]["n_cells"], 12)
            self.assertEqual(inspect_run(run)["integrity"], [])
            report = write_report([run], root / "report.md")
            text = report.read_text()
            self.assertIn("All inputs in this report are generated synthetic fixtures", text)
            self.assertIn("Disease prediction", text)
            with self.assertRaises(FileExistsError):
                write_report([run], report)

    def test_tampered_summary_counts_are_withheld(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            (run / "summary.json").write_text('{"counts":{"n_cells":999}}')
            inspected = inspect_run(run)
            self.assertEqual(inspected["counts"], {})
            self.assertTrue(inspected["integrity"])

    def test_missing_and_failed_runs_remain_visible_and_html_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "failed"
            run.mkdir()
            write_json(run / "run.json", {"status": "<script>alert(1)</script>", "data_kind": "synthetic", "failure_type": "ValueError"})
            report = write_report([run, root / "missing"], root / "report.html", output_format="html")
            text = report.read_text()
            self.assertIn("&lt;script&gt;", text)
            self.assertNotIn("<script>", text)
            self.assertIn("missing_run_record", text)
            self.assertIn("missing or unreadable", text)
            self.assertNotIn("https://", text)


_CORE_AVAILABLE = all(importlib.util.find_spec(f"ftd_prediction.{name}") for name in ("contracts", "data", "rna", "spatial"))


@unittest.skipUnless(_CORE_AVAILABLE, "core/spatial modules are integrated by the lead before end-to-end checks")
class IntegratedRunnerTests(unittest.TestCase):
    def _change_cells(self, manifest, transform):
        path = manifest.parent / "cells.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        transform(rows)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_preflight_runs_both_assays_and_keeps_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(Path(tmp) / "unique_preflight")
            status = json.loads((result / "preflight.json").read_text())
            self.assertEqual(status["status"], "complete")
            for name in ("rna", "spatial"):
                record = json.loads((result / name / "run.json").read_text())
                self.assertTrue(record["inputs_unchanged"])
                self.assertFalse(record["disease_inference_enabled"])
                self.assertTrue(record["code"]["source_sha256"])
                self.assertEqual(inspect_run(result / name)["integrity"], [])
            self.assertTrue((result / "report.html").is_file())

    def test_all_isolate_neighborhoods_are_unavailable_not_zero_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_synthetic_fixtures(root / "fixtures")["spatial"]
            settings = load_config(None)
            settings["radius_um"] = 0.001
            output = run_pilot(manifest, root / "isolates", config=settings)
            spatial = json.loads((output / "spatial.json").read_text())
            self.assertEqual(spatial["isolate_fraction"], 1.0)
            for row in spatial["pooled_neighborhoods"]:
                self.assertEqual(row["status"], "unavailable_no_neighbors")
                self.assertIsNone(row["mean"])
            self.assertEqual(spatial["conditional_null"]["standardized_residual_available_fraction"], 0.0)

    def test_missing_cell_types_preserve_graph_and_unavailable_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_synthetic_fixtures(root / "fixtures")["spatial"]
            self._change_cells(manifest, lambda rows: [row.update(cell_type="") for row in rows])
            output = run_pilot(manifest, root / "unlabelled")
            spatial = json.loads((output / "spatial.json").read_text())
            self.assertGreater(spatial["edge_count"], 0)
            self.assertEqual(spatial["neighborhoods"]["status"], "unavailable")
            self.assertEqual(spatial["conditional_null"]["reason"], "cell_type_annotations_missing")
            self.assertTrue(all(row["mean"] is None for row in spatial["pooled_neighborhoods"]))

    def test_partial_component_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_synthetic_fixtures(root / "fixtures")["spatial"]
            self._change_cells(manifest, lambda rows: rows[0].update(tissue_component=""))
            with self.assertRaises(ValueError):
                run_pilot(manifest, root / "partial_components")
            record = json.loads((root / "partial_components" / "run.json").read_text())
            self.assertEqual(record["status"], "failed")

    def test_missing_regions_preserve_sections_without_merging_unknown_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_synthetic_fixtures(root / "fixtures")["spatial"]
            payload = json.loads(manifest.read_text())
            for record in payload["records"]:
                record["retinal_region"] = None
            manifest.write_text(json.dumps(payload))
            output = run_pilot(manifest, root / "unknown_regions")
            spatial = json.loads((output / "spatial.json").read_text())
            self.assertGreater(spatial["edge_count"], 0)
            self.assertEqual(len(spatial["pooled_neighborhoods"]), len(payload["records"]))
            for row in spatial["pooled_neighborhoods"]:
                self.assertEqual(row["status"], "unavailable_region_comparability")
                self.assertIsNone(row["mean"])
                self.assertEqual(len(row["record_ids"]), 1)

    def test_manifest_scope_or_input_change_after_validation_is_not_loaded(self):
        for field, value in (("data_kind", "study"), ("counts_file", "unapproved-counts.csv")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = generate_synthetic_fixtures(root / "fixtures")["rna"]
                def mutate():
                    payload = json.loads(manifest.read_text())
                    payload[field] = value
                    manifest.write_text(json.dumps(payload))
                    return {"source_sha256": "synthetic_test"}
                with patch("ftd_prediction.cli.source_snapshot", side_effect=mutate), patch("ftd_prediction.data._read_cells") as cells_reader:
                    with self.assertRaises(RuntimeError):
                        run_pilot(manifest, root / "changed")
                    cells_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
