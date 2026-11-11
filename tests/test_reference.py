import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ftd_prediction.contracts import ContractError
from ftd_prediction.reference import CentroidReference


def model():
    return CentroidReference.fit(
        np.array([[10, 0], [0, 10], [9, 1], [1, 9], [3, 2]], dtype=np.int64),
        ["G1", "G2"], ["RGC", "Rod", "RGC", "Rod", "heldout-only-type"],
        ["fit-a", "fit-a", "fit-b", "fit-b", "test-a"],
        fit_donors=["fit-a", "fit-b"],
    )


class ReferenceTests(unittest.TestCase):
    def test_fit_excludes_test_donor_and_label(self):
        reference = model()
        self.assertEqual(reference.categories, ("RGC", "Rod"))
        self.assertEqual(reference.provenance["fit_donors"], ["fit-a", "fit-b"])
        self.assertEqual(reference.provenance["n_fitted_rows"], 4)
        self.assertEqual(reference.provenance["n_excluded_donor_rows"], 1)

    def test_heldout_counts_and_labels_cannot_change_fit(self):
        first = CentroidReference.fit(np.array([[10, 0], [0, 10], [1, 1]]), ["G1", "G2"],
                                      ["RGC", "Rod", "ignored-a"], ["fit", "fit", "test"], fit_donors=["fit"])
        second = CentroidReference.fit(np.array([[10, 0], [0, 10], [np.nan, np.inf]]), ["G1", "G2"],
                                       ["RGC", "Rod", "ignored-b"], ["fit", "fit", "test"], fit_donors=["fit"])
        np.testing.assert_array_equal(first.centroids, second.centroids)
        self.assertEqual(first.categories, second.categories)

    def test_inference_is_independent_of_other_query_cells(self):
        reference = model()
        alone = reference.predict([[8, 1]], ["G1", "G2"])
        combined = reference.predict([[8, 1], [10000, 1], [0, 10000]], ["G1", "G2"])
        self.assertEqual(alone["labels"][0], combined["labels"][0])
        self.assertEqual(alone["distances"][0], combined["distances"][0])
        self.assertEqual(alone["scores"][0], combined["scores"][0])

    def test_gene_alignment_is_explicit_and_extra_genes_do_not_change_inference(self):
        reference = model()
        original = reference.predict([[8, 1]], ["G1", "G2"])
        reordered = reference.predict([[1, 9999, 8]], ["G2", "extra", "G1"])
        self.assertEqual(original["distances"], reordered["distances"])
        for genes, counts in ((["G1"], [[8]]), (["G1", "G1"], [[8, 1]])):
            with self.subTest(genes=genes), self.assertRaises(ContractError):
                reference.predict(counts, genes)

    def test_donor_balancing_is_not_cell_count_weighting(self):
        a = np.array([[10, 0], [0, 10]], dtype=np.int64)
        first = CentroidReference.fit(a, ["G1", "G2"], ["RGC", "RGC"], ["a", "b"], fit_donors=["a", "b"])
        more_a = np.array([[10, 0], [10, 0], [10, 0], [0, 10]], dtype=np.int64)
        second = CentroidReference.fit(more_a, ["G1", "G2"], ["RGC"] * 4, ["a", "a", "a", "b"], fit_donors=["a", "b"])
        np.testing.assert_allclose(first.centroids, second.centroids)

    def test_missing_annotations_are_recorded_and_excluded(self):
        reference = CentroidReference.fit([[10, 0], [1, 1]], ["G1", "G2"], ["RGC", None], ["a", "a"], fit_donors=["a"])
        self.assertEqual(reference.provenance["n_missing_annotation_rows"], 1)
        self.assertEqual(reference.provenance["n_fitted_rows"], 1)

    def test_zero_library_query_has_missing_label(self):
        result = model().predict([[0, 0]], ["G1", "G2"])
        self.assertEqual(result["labels"], [None])
        self.assertEqual(result["zero_library"], [True])
        self.assertEqual(result["evidence_scope"], "cell_annotation_reference")
        self.assertNotIn("probabilities", result)

    def test_model_parameters_and_inputs_remain_unchanged(self):
        reference = model()
        before = reference.centroids.copy()
        provenance = copy.deepcopy(reference.provenance)
        query = np.array([[8, 1]], dtype=np.int64)
        reference.predict(query, ["G1", "G2"])
        np.testing.assert_array_equal(reference.centroids, before)
        self.assertEqual(reference.provenance, provenance)
        np.testing.assert_array_equal(query, [[8, 1]])
        with self.assertRaises(ValueError):
            reference.centroids[0, 0] = 0

    def test_save_load_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation-reference.json"
            reference = model()
            reference.save(path)
            recovered = CentroidReference.load(path)
            self.assertEqual(reference.predict([[8, 1]], ["G1", "G2"]), recovered.predict([[8, 1]], ["G1", "G2"]))
            self.assertEqual(reference.provenance, recovered.provenance)
            with self.assertRaises(FileExistsError):
                reference.save(path)

    def test_corrupt_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation-reference.json"
            model().save(path)
            payload = json.loads(path.read_text())
            payload["centroids"][0][0] = float("nan")
            path.write_text(json.dumps(payload))
            with self.assertRaises(ContractError):
                CentroidReference.load(path)

    def test_no_fitting_cells_or_zero_library_rejected(self):
        for donors, counts in ((["test"], [[10, 0]]), (["fit"], [[0, 0]])):
            with self.subTest(donors=donors), self.assertRaises(ContractError):
                CentroidReference.fit(counts, ["G1", "G2"], ["RGC"], donors, fit_donors=["fit"])

    def test_malformed_raw_query_and_normalization_rejected(self):
        reference = model()
        for counts in ([[0.5, 1]], [[-1, 1]], [[float("nan"), 1]], [[2**64, 1]]):
            with self.subTest(counts=counts), self.assertRaises(ContractError):
                reference.predict(counts, ["G1", "G2"])
        for total in (0, -1, True, float("inf")):
            with self.subTest(total=total), self.assertRaises(ContractError):
                CentroidReference.fit([[1, 0]], ["G1", "G2"], ["RGC"], ["fit"], fit_donors=["fit"], normalization_total=total)

    def test_all_missing_annotation_tokens_excluded_from_reference_classes(self):
        missing = [None, "", "unknown", "unresolved", "restricted", "not_applicable", " UNKNOWN ", "   "]
        labels = ["RGC", *missing]
        reference = CentroidReference.fit([[10, 0]] * len(labels), ["G1", "G2"], labels,
                                          ["fit"] * len(labels), fit_donors=["fit"])
        self.assertEqual(reference.categories, ("RGC",))
        self.assertEqual(reference.provenance["n_fitted_rows"], 1)
        self.assertEqual(reference.provenance["n_missing_annotation_rows"], len(missing))

    def test_missing_token_cannot_be_loaded_as_reference_category(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            model().save(path)
            payload = json.loads(path.read_text())
            for token in ("unknown", "unresolved", "restricted", "not_applicable"):
                payload["categories"][0] = token
                path.write_text(json.dumps(payload))
                with self.subTest(token=token), self.assertRaises(ContractError):
                    CentroidReference.load(path)

    def test_minimal_or_internally_inconsistent_provenance_rejected(self):
        mutations = [
            lambda payload: payload.update(provenance={"evidence_scope": "cell_annotation_reference", "fit_donors": ["made-up-donor"]}),
            lambda payload: payload["provenance"].update(n_input_rows=99),
            lambda payload: payload["provenance"].update(n_fitted_rows=True),
            lambda payload: payload["provenance"].update(n_excluded_donor_rows=-1),
            lambda payload: payload["provenance"].update(normalization="different-normalization"),
            lambda payload: payload["provenance"].update(normalization_total=42.0),
            lambda payload: payload["provenance"].update(fit_donors=["outside-requested-donors"]),
            lambda payload: payload["provenance"].update(schema_version=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            model().save(path)
            original = json.loads(path.read_text())
            for index, mutate in enumerate(mutations):
                payload = copy.deepcopy(original)
                mutate(payload)
                path.write_text(json.dumps(payload))
                with self.subTest(mutation=index), self.assertRaises(ContractError):
                    CentroidReference.load(path)

    def test_centroids_cannot_exceed_fitted_normalization_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            model().save(path)
            payload = json.loads(path.read_text())
            payload["centroids"][0][0] = 1e300
            path.write_text(json.dumps(payload))
            with self.assertRaises(ContractError):
                CentroidReference.load(path)

    def test_nonfinite_distance_guard_catches_invalid_in_memory_parameters(self):
        reference = model()
        invalid = replace(reference, centroids=np.full_like(reference.centroids, 1e300))
        with self.assertRaises(ContractError):
            invalid.predict([[8, 1]], ["G1", "G2"])


if __name__ == "__main__":
    unittest.main()
