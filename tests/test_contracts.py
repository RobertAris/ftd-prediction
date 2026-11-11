import copy
import unittest

from ftd_prediction.contracts import (
    ContractError, require_graph_ready, validate_donor_splits, validate_manifest,
)


def record(**overrides):
    value = {
        "record_id": "record-a", "assay": "xenium", "donor_id": "donor-a",
        "sample_id": "sample-a", "retinal_region": "peripheral",
        "processing_batch": "batch-a", "source_label": "FTD",
        "diagnosis_status": "provisional", "label_evidence_ref": "synthetic-label-v1",
        "mapping_status": "confirmed", "mapping_evidence_ref": "synthetic-map-v1",
        "section_id": "section-a", "acquisition_region_id": "acquisition-a",
        "panel_ref": "synthetic-panel-v1", "segmentation_ref": "synthetic-seg-v1",
        "coordinate_units": "um", "coordinate_transform_ref": "identity-v1",
    }
    value.update(overrides)
    return value


def manifest(records=None):
    return {
        "schema_version": 1, "data_kind": "synthetic", "dataset_id": "synthetic-v1",
        "source_version": "1", "counts_file": "counts.csv", "cells_file": "cells.csv",
        "records": records if records is not None else [record()],
    }


class ContractTests(unittest.TestCase):
    def test_validation_preserves_source_and_does_not_modify_input(self):
        payload = manifest()
        before = copy.deepcopy(payload)
        result = validate_manifest(payload)
        result["records"][0]["sample_id"] = "changed-copy"
        self.assertEqual(payload, before)
        self.assertEqual(result["records"][0]["source_label"], "FTD")

    def test_multiple_sections_in_one_acquisition_are_valid(self):
        payload = manifest([record(), record(record_id="record-b", section_id="section-b")])
        require_graph_ready(payload)

    def test_preserves_explicit_unresolved_inventory_but_blocks_graph(self):
        payload = manifest([record(mapping_status="unresolved", section_id="unknown")])
        self.assertEqual(validate_manifest(payload)["records"][0]["section_id"], "unknown")
        with self.assertRaises(ContractError):
            require_graph_ready(payload)

    def test_missing_spatial_inventory_fields_block_graph(self):
        for field in ("section_id", "panel_ref", "segmentation_ref", "coordinate_transform_ref"):
            with self.subTest(field=field):
                payload = manifest()
                del payload["records"][0][field]
                validate_manifest(payload)
                with self.assertRaises(ContractError):
                    require_graph_ready(payload)

    def test_pixel_units_are_not_silently_converted(self):
        with self.assertRaises(ContractError):
            require_graph_ready(manifest([record(coordinate_units="pixels")]))

    def test_rna_manifest_is_not_graph_ready(self):
        with self.assertRaises(ContractError):
            require_graph_ready(manifest([record(assay="chromium_snrna", section_id=None)]))

    def test_duplicate_record_ids_rejected(self):
        with self.assertRaises(ContractError):
            validate_manifest(manifest([record(), record(section_id="section-b")]))

    def test_duplicate_or_contradictory_physical_sections_rejected(self):
        for donor in ("donor-a", "donor-b"):
            with self.subTest(donor=donor), self.assertRaises(ContractError):
                validate_manifest(manifest([record(), record(record_id="record-b", donor_id=donor)]))

    def test_reprocessed_rna_sample_cannot_be_counted_twice(self):
        first = record(assay="chromium_snrna", section_id=None)
        second = record(record_id="record-b", assay="chromium_snrna", section_id=None)
        with self.assertRaises(ContractError):
            validate_manifest(manifest([first, second]))

    def test_same_donor_all_assays_stays_in_one_split(self):
        first = record(split="train")
        second = record(record_id="record-b", assay="chromium_snrna", sample_id="sample-b", section_id=None, split="test")
        with self.assertRaises(ContractError):
            validate_manifest(manifest([first, second]))

    def test_distinct_donors_can_have_distinct_splits(self):
        validate_donor_splits([record(split="train"), record(donor_id="donor-b", split="test")])

    def test_no_auto_split_or_label_harmonization(self):
        payload = manifest([record(), record(record_id="record-b", donor_id="donor-b", sample_id="sample-b", section_id="section-b", source_label="ND")])
        output = validate_manifest(payload)
        self.assertEqual([entry["split"] for entry in output["records"]], [None, None])
        self.assertEqual([entry["source_label"] for entry in output["records"]], ["FTD", "ND"])

    def test_same_assay_donor_label_conflict_rejected(self):
        with self.assertRaises(ContractError):
            validate_manifest(manifest([record(), record(record_id="record-b", section_id="section-b", source_label="ND")]))

    def test_unresolved_donor_not_assigned_a_split(self):
        with self.assertRaises(ContractError):
            validate_manifest(manifest([record(donor_id="unknown", split="test")]))

    def test_missing_required_fields_and_control_characters(self):
        for field in ("dataset_id", "source_version", "counts_file", "cells_file"):
            payload = manifest()
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_manifest(payload)
        with self.assertRaises(ContractError):
            validate_manifest(manifest([record(donor_id="private\nvalue")]))

    def test_bad_schema_types_raise_contract_errors(self):
        for value in (True, 1.0, "1", None):
            payload = manifest()
            payload["schema_version"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_manifest(payload)

    def test_malformed_choice_types_raise_contract_errors(self):
        for value in ([], {}, True, None):
            payload = manifest()
            payload["data_kind"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_manifest(payload)
        for value in ([], {}, True):
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_manifest(manifest([record(split=value)]))

    def test_errors_do_not_disclose_identifier(self):
        private = "private-synthetic-token"
        payload = manifest([record(record_id=private), record(record_id=private, section_id="section-b")])
        with self.assertRaises(ContractError) as caught:
            validate_manifest(payload)
        self.assertNotIn(private, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
