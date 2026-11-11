import json
from pathlib import Path
import unittest

import numpy as np

from ftd_prediction.contracts import ContractError
from ftd_prediction.data import CellDataset
from ftd_prediction.rna import summarize_expression


def dataset(counts=None, *, unresolved=False):
    records = []
    for index, (donor, region) in enumerate((("donor-a", "peripheral"), ("donor-a", "macular"), ("donor-b", "peripheral"))):
        records.append({
            "record_id": f"record-{index}", "assay": "chromium_snrna", "donor_id": donor,
            "sample_id": f"sample-{index}", "retinal_region": region, "processing_batch": "batch-a",
            "source_label": "FTLD" if donor == "donor-a" else "Healthy", "diagnosis_status": "confirmed",
            "label_evidence_ref": "synthetic-v1", "mapping_status": "unresolved" if unresolved else "confirmed",
            "mapping_evidence_ref": "synthetic-v1",
        })
    manifest = {"schema_version": 1, "data_kind": "synthetic", "dataset_id": "fixture", "source_version": "v1",
                "counts_file": "counts.csv", "cells_file": "cells.csv", "records": records}
    cells = [
        {"cell_id": "a", "record_id": "record-0", "cell_type": "RGC"},
        {"cell_id": "b", "record_id": "record-0", "cell_type": "RGC"},
        {"cell_id": "c", "record_id": "record-1", "cell_type": "RGC"},
        {"cell_id": "d", "record_id": "record-2", "cell_type": None},
    ]
    matrix = np.array([[1, 0], [0, 3], [5, 1], [0, 0]], dtype=np.int64) if counts is None else counts
    return CellDataset(matrix, ("G1", "G2"), cells, manifest, (Path("synthetic"),))


class ExpressionTests(unittest.TestCase):
    def test_exact_raw_pseudobulk_keeps_region_and_donor(self):
        summary = summarize_expression(dataset())
        self.assertEqual(summary["genes"], ["G1", "G2"])
        self.assertEqual(len(summary["rows"]), 3)
        rows = {(row["donor_id"], row["retinal_region"]): row for row in summary["rows"]}
        peripheral = rows[("donor-a", "peripheral")]
        self.assertEqual(peripheral["raw_counts"], [1, 3])
        self.assertEqual(peripheral["n_cells"], 2)
        self.assertEqual(peripheral["source_labels"], ["FTLD"])
        np.testing.assert_allclose(peripheral["log1p_normalized"], np.log1p([2500, 7500]))
        self.assertEqual(rows[("donor-a", "macular")]["raw_counts"], [5, 1])

    def test_missing_annotation_remains_flagged(self):
        summary = summarize_expression(dataset())
        missing = [row for row in summary["rows"] if row["cell_type_missing"]]
        self.assertEqual(len(missing), 1)
        self.assertIsNone(missing[0]["cell_type"])
        self.assertEqual(missing[0]["log1p_normalized"], [None, None])
        self.assertEqual(missing[0]["library_total"], 0)
        self.assertTrue(missing[0]["zero_library"])
        self.assertFalse(any(row["cell_type"] == "Rod" for row in summary["rows"]))
        json.dumps(summary, allow_nan=False)

    def test_pseudobulk_sum_does_not_wrap_int64(self):
        high = np.iinfo(np.int64).max
        counts = np.array([[high, 0], [high, 0], [5, 1], [0, 0]], dtype=np.int64)
        result = summarize_expression(dataset(counts))
        row = next(row for row in result["rows"] if row["donor_id"] == "donor-a" and row["retinal_region"] == "peripheral")
        self.assertEqual(row["raw_counts"], [2 * int(high), 0])
        self.assertIs(type(row["raw_counts"][0]), int)

    def test_unresolved_donor_mapping_blocks_summary(self):
        with self.assertRaises(ContractError):
            summarize_expression(dataset(unresolved=True))

    def test_invalid_raw_counts_block_summary(self):
        for matrix in (np.ones((4, 2), dtype=float), -np.ones((4, 2), dtype=np.int64)):
            with self.subTest(dtype=matrix.dtype), self.assertRaises(ContractError):
                summarize_expression(dataset(matrix))

    def test_summary_does_not_mutate_count_matrix(self):
        data = dataset()
        before = data.counts.copy()
        summarize_expression(data)
        np.testing.assert_array_equal(data.counts, before)

    def test_unknown_regions_keep_separate_record_summaries(self):
        for region in ("unknown", "unresolved", "restricted", "not_applicable", "", None):
            with self.subTest(region=region):
                data = dataset()
                data.manifest["records"][0]["retinal_region"] = region
                data.manifest["records"][1]["retinal_region"] = region
                rows = [row for row in summarize_expression(data)["rows"] if row["donor_id"] == "donor-a"]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["pooling_scope"] == "record" and not row["region_resolved"] for row in rows))
                self.assertEqual({tuple(row["record_ids"]) for row in rows}, {("record-0",), ("record-1",)})
                self.assertEqual(sorted(row["raw_counts"] for row in rows), [[1, 3], [5, 1]])

    def test_known_regions_can_pool_records_with_provenance(self):
        data = dataset()
        data.manifest["records"][1]["retinal_region"] = "peripheral"
        rows = [row for row in summarize_expression(data)["rows"] if row["donor_id"] == "donor-a"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pooling_scope"], "donor_region")
        self.assertTrue(rows[0]["region_resolved"])
        self.assertEqual(rows[0]["record_ids"], ["record-0", "record-1"])
        self.assertEqual(rows[0]["raw_counts"], [6, 4])
        self.assertEqual(rows[0]["library_total"], 10)
        self.assertFalse(rows[0]["zero_library"])

    def test_missing_region_field_does_not_invent_comparability(self):
        data = dataset()
        del data.manifest["records"][0]["retinal_region"]
        row = next(row for row in summarize_expression(data)["rows"] if row["record_ids"] == ["record-0"])
        self.assertIsNone(row["retinal_region"])
        self.assertEqual(row["pooling_scope"], "record")

    def test_direct_dataset_missing_annotations_are_not_treated_as_cell_types(self):
        data = dataset()
        data.cells[0]["cell_type"] = "restricted"
        data.cells[1]["cell_type"] = "not_applicable"
        rows = summarize_expression(data)["rows"]
        row = next(row for row in rows if row["record_ids"] == ["record-0"])
        self.assertIsNone(row["cell_type"])
        self.assertTrue(row["cell_type_missing"])
        self.assertEqual(row["n_cells"], 2)


if __name__ == "__main__":
    unittest.main()
