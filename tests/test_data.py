import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ftd_prediction.contracts import ContractError
from ftd_prediction.data import load_dataset, resolved_cells


def fixture(root, *, assay="chromium_snrna"):
    record = {
        "record_id": "record-a", "assay": assay, "donor_id": "donor-a",
        "sample_id": "sample-a", "retinal_region": "peripheral", "processing_batch": "batch-a",
        "source_label": "Healthy", "diagnosis_status": "confirmed", "label_evidence_ref": "synthetic-v1",
        "mapping_status": "confirmed", "mapping_evidence_ref": "synthetic-v1",
    }
    if assay == "xenium":
        record.update(section_id="section-a", acquisition_region_id="acquisition-a", panel_ref="panel-v1",
                      segmentation_ref="seg-v1", coordinate_units="um", coordinate_transform_ref="identity-v1")
    manifest = {"schema_version": 1, "data_kind": "synthetic", "dataset_id": "fixture", "source_version": "v1",
                "counts_file": "counts.csv", "cells_file": "cells.csv", "records": [record]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with (root / "cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "record_id", "cell_type", "layer", "x", "y", "tissue_component"])
        writer.writerow(["cell-a", "record-a", "RGC", "GCL" if assay == "xenium" else "", "1" if assay == "xenium" else "", "2" if assay == "xenium" else "", ""])
        writer.writerow(["cell-b", "record-a", "", "GCL" if assay == "xenium" else "", "3" if assay == "xenium" else "", "4" if assay == "xenium" else "", ""])
    (root / "counts.csv").write_text("cell_id,G1,G2\ncell-b,3,4\ncell-a,1,2\n")
    return root / "manifest.json"


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_aligns_by_cell_id_and_keeps_inputs_immutable(self):
        paths = [self.root / name for name in ("manifest.json", "cells.csv", "counts.csv")]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        dataset = load_dataset(self.manifest)
        np.testing.assert_array_equal(dataset.counts, [[1, 2], [3, 4]])
        self.assertEqual(dataset.genes, ("G1", "G2"))
        self.assertIsNone(dataset.cells[1]["cell_type"])
        self.assertEqual(set(path.resolve() for path in paths), set(dataset.input_paths))
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        with self.assertRaises(ValueError):
            dataset.counts[0, 0] = 99

    def test_resolved_xenium_identity_is_physical_section(self):
        fixture(self.root, assay="xenium")
        cells = resolved_cells(load_dataset(self.manifest))
        self.assertEqual(cells[0]["section_key"], ("donor-a", "sample-a", "section-a"))
        self.assertEqual(cells[0]["x"], 1.0)

    def test_invalid_counts_rejected(self):
        for value in ("-1", "0.5", "nan", "inf", "", str(2**63), "not-a-count"):
            with self.subTest(value=value):
                (self.root / "counts.csv").write_text(f"cell_id,G1,G2\ncell-a,{value},2\ncell-b,3,4\n")
                with self.assertRaises(ContractError):
                    load_dataset(self.manifest)

    def test_exact_integer_decimal_counts_are_supported(self):
        (self.root / "counts.csv").write_text("cell_id,G1,G2\ncell-a,1.0,2e0\ncell-b,3,4\n")
        np.testing.assert_array_equal(load_dataset(self.manifest).counts, [[1, 2], [3, 4]])

    def test_invalid_count_row_shapes_and_identifiers(self):
        contents = [
            "cell_id,G1,G2\ncell-a,1\ncell-b,3,4\n",
            "cell_id,G1,G2\ncell-a,1,2\ncell-a,3,4\n",
            "cell_id,G1,G2\ncell-a,1,2\ncell-z,3,4\n",
            "cell_id,G1,G2\ncell-a,1,2\n",
            "cell_id,G1,G1\ncell-a,1,2\ncell-b,3,4\n",
        ]
        for content in contents:
            with self.subTest(content=content):
                (self.root / "counts.csv").write_text(content)
                with self.assertRaises(ContractError):
                    load_dataset(self.manifest)

    def test_bounds_checked_before_dense_allocation(self):
        with self.assertRaises(ContractError):
            load_dataset(self.manifest, max_cells=1)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest, max_genes=1)
        with patch("ftd_prediction.data.MAX_DENSE_ELEMENTS", 3), patch("ftd_prediction.data.np.empty") as allocator:
            with self.assertRaises(ContractError):
                load_dataset(self.manifest)
            allocator.assert_not_called()

    def test_nonpositive_or_boolean_limits_rejected(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ContractError):
                load_dataset(self.manifest, max_cells=limit)

    def test_duplicate_cells_rejected(self):
        content = (self.root / "cells.csv").read_text().replace("cell-b", "cell-a")
        (self.root / "cells.csv").write_text(content)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest)

    def test_unknown_record_and_conflicting_cell_metadata_rejected(self):
        cells_path = self.root / "cells.csv"
        original = cells_path.read_text()
        cells_path.write_text(original.replace("record-a", "record-missing"))
        with self.assertRaises(ContractError):
            load_dataset(self.manifest)
        rows = list(csv.reader(original.splitlines()))
        rows[0].append("donor_id")
        rows[1].append("wrong-donor")
        rows[2].append("wrong-donor")
        with cells_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest)

    def test_nonfinite_coordinates_rejected(self):
        fixture(self.root, assay="xenium")
        content = (self.root / "cells.csv").read_text().replace(",1,2,", ",nan,2,")
        (self.root / "cells.csv").write_text(content)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest)

    def test_missing_xenium_coordinates_rejected(self):
        fixture(self.root, assay="xenium")
        content = (self.root / "cells.csv").read_text().replace(",1,2,", ",,,")
        (self.root / "cells.csv").write_text(content)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest)

    def test_duplicate_manifest_json_keys_rejected(self):
        text = self.manifest.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        self.manifest.write_text(text)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest)
        with self.assertRaises(ContractError):
            load_dataset(self.manifest, manifest_bytes=self.manifest.read_bytes())

    def test_checked_manifest_snapshot_rejects_live_scope_or_path_changes_before_assay_reads(self):
        for field, value in (("data_kind", "study"), ("counts_file", "different-counts.csv")):
            fixture(self.root)
            checked = self.manifest.read_bytes()
            changed = json.loads(checked)
            changed[field] = value
            self.manifest.write_text(json.dumps(changed))
            with self.subTest(field=field), patch("ftd_prediction.data._read_cells") as cells_reader, patch("ftd_prediction.data._read_counts") as count_reader:
                with self.assertRaises(ContractError):
                    load_dataset(self.manifest, manifest_bytes=checked)
                cells_reader.assert_not_called()
                count_reader.assert_not_called()

    def test_checked_input_resolution_rejects_changed_symlink_before_assay_reads(self):
        linked = self.root / "linked-counts.csv"
        linked.symlink_to("counts.csv")
        payload = json.loads(self.manifest.read_bytes())
        payload["counts_file"] = linked.name
        self.manifest.write_text(json.dumps(payload))
        checked = self.manifest.read_bytes()
        expected = (self.manifest.resolve(), (self.root / "counts.csv").resolve(), (self.root / "cells.csv").resolve())
        linked.unlink()
        linked.symlink_to("other-counts.csv")
        with patch("ftd_prediction.data._read_cells") as cells_reader, patch("ftd_prediction.data._read_counts") as count_reader:
            with self.assertRaises(ContractError):
                load_dataset(self.manifest, manifest_bytes=checked, expected_input_paths=expected)
            cells_reader.assert_not_called()
            count_reader.assert_not_called()

    def test_unchanged_checked_snapshot_is_consumed(self):
        checked = self.manifest.read_bytes()
        expected = (self.manifest.resolve(), (self.root / "counts.csv").resolve(), (self.root / "cells.csv").resolve())
        dataset = load_dataset(self.manifest, manifest_bytes=checked, expected_input_paths=expected)
        np.testing.assert_array_equal(dataset.counts, [[1, 2], [3, 4]])
        self.assertEqual(dataset.input_paths, expected)

    def test_symlinked_input_files_resolve_for_provenance(self):
        linked = self.root / "linked-counts.csv"
        linked.symlink_to("counts.csv")
        payload = json.loads(self.manifest.read_text())
        payload["counts_file"] = "linked-counts.csv"
        self.manifest.write_text(json.dumps(payload))
        self.assertEqual(load_dataset(self.manifest).input_paths[1], (self.root / "counts.csv").resolve())

    def test_all_annotation_missing_tokens_are_normalized(self):
        original = (self.root / "cells.csv").read_text()
        for token in ("unknown", "unresolved", "restricted", "not_applicable", "", "UNKNOWN", "   "):
            with self.subTest(token=token):
                (self.root / "cells.csv").write_text(original.replace("RGC", token))
                self.assertIsNone(load_dataset(self.manifest).cells[0]["cell_type"])


if __name__ == "__main__":
    unittest.main()
