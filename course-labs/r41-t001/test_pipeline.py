"""Unit tests for R41-T001 schema validator and synthetic fixture.

Uses Python's standard library `unittest` framework to verify:
- Valid fixture passes completely.
- Detection of missing, empty, or duplicate record IDs.
- Detection of missing or empty group IDs.
- Strict rejection of malformed numeric values, booleans, and sequences/tensors.
- Strict rejection of non-integer or out-of-range binary labels.
- Reporting of multiple failures within a single record without silent dropping.
- Verification of failure report fields (record_id, field, reason).
- Verification of the fixed stratified split and synthetic data disclaimer.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure local modules (pipeline, fixture) are importable regardless of current working directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fixture import (  # noqa: E402
    DATASET_DISCLAIMER,
    STRATIFIED_SPLIT,
    SYNTHETIC_FIXTURE,
    get_stratified_split,
    get_synthetic_fixture,
)
from pipeline import (  # noqa: E402
    DatasetValidationResult,
    ValidationFailure,
    validate_dataset,
    validate_single_record,
)


class TestSyntheticFixtureSpecification(unittest.TestCase):
    """Test conformity of the synthetic fixture with Issue #1 specification."""

    def test_fixture_row_count(self) -> None:
        fixture = get_synthetic_fixture()
        self.assertEqual(len(fixture), 8, "Fixture must contain exactly eight rows.")

    def test_fixture_exact_field_values(self) -> None:
        fixture = get_synthetic_fixture()
        expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]

        actual_x1 = [row["x1"] for row in fixture]
        actual_x2 = [row["x2"] for row in fixture]
        actual_labels = [row["label"] for row in fixture]

        self.assertEqual(actual_x1, expected_x1)
        self.assertEqual(actual_x2, [1] * 8)
        self.assertEqual(actual_labels, expected_labels)

    def test_fixture_ids_are_stable_unique_strings(self) -> None:
        fixture = get_synthetic_fixture()
        record_ids = [row["record_id"] for row in fixture]
        group_ids = [row["group_id"] for row in fixture]

        self.assertEqual(len(set(record_ids)), 8, "All record_ids must be unique.")
        for rid in record_ids:
            self.assertIsInstance(rid, str)
            self.assertTrue(bool(rid.strip()))

        for gid in group_ids:
            self.assertIsInstance(gid, str)
            self.assertTrue(bool(gid.strip()))

    def test_stratified_split_integrity(self) -> None:
        split = get_stratified_split()
        train_rows = split["train"]
        test_rows = split["test"]

        self.assertEqual(len(train_rows), 6)
        self.assertEqual(len(test_rows), 2)

        train_labels = [r["label"] for r in train_rows]
        test_labels = [r["label"] for r in test_rows]

        # 50/50 balance in both splits (stratified)
        self.assertEqual(train_labels.count(0), 3)
        self.assertEqual(train_labels.count(1), 3)
        self.assertEqual(test_labels.count(0), 1)
        self.assertEqual(test_labels.count(1), 1)

        # No data leakage / disjoint record IDs
        train_ids = {r["record_id"] for r in train_rows}
        test_ids = {r["record_id"] for r in test_rows}
        self.assertTrue(train_ids.isdisjoint(test_ids))

    def test_disclaimer_present(self) -> None:
        self.assertIn("Synthetic", DATASET_DISCLAIMER)
        self.assertIn("NOT represent real IPL", DATASET_DISCLAIMER)


class TestValidFixturePasses(unittest.TestCase):
    """Verify that the valid synthetic fixture passes schema validation cleanly."""

    def test_canonical_fixture_passes_validation(self) -> None:
        fixture = get_synthetic_fixture()
        result = validate_dataset(fixture)

        self.assertTrue(result.all_valid)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.valid_count, 8)
        self.assertEqual(result.invalid_count, 0)
        self.assertEqual(result.total_records, 8)

    def test_empty_dataset_handling(self) -> None:
        result = validate_dataset([])
        self.assertTrue(result.all_valid)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.total_records, 0)


class TestRecordIdValidation(unittest.TestCase):
    """Test schema enforcement on 'record_id'."""

    def test_missing_record_id(self) -> None:
        bad_record = {"group_id": "grp_01", "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "record_id"]
        self.assertGreaterEqual(len(failures), 1)
        self.assertIn("Missing", failures[0].reason)

    def test_none_record_id(self) -> None:
        bad_record = {"record_id": None, "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "record_id"]
        self.assertGreaterEqual(len(failures), 1)
        self.assertIn("cannot be None", failures[0].reason)

    def test_empty_string_record_id(self) -> None:
        bad_record = {"record_id": "   ", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "record_id"]
        self.assertGreaterEqual(len(failures), 1)
        self.assertIn("empty or whitespace", failures[0].reason)

    def test_non_string_record_id(self) -> None:
        bad_record = {"record_id": 12345, "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "record_id"]
        self.assertGreaterEqual(len(failures), 1)
        self.assertIn("must be a string", failures[0].reason)

    def test_duplicate_record_id(self) -> None:
        records = [
            {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 0},
            {"record_id": "rec_001", "group_id": "grp_02", "x1": 4, "x2": 1, "label": 1},
        ]
        result = validate_dataset(records)

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "record_id"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].record_id, "rec_001")
        self.assertIn("Duplicate record_id", failures[0].reason)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.invalid_count, 1)


class TestGroupIdValidation(unittest.TestCase):
    """Test schema enforcement on 'group_id'."""

    def test_missing_group_id(self) -> None:
        bad_record = {"record_id": "rec_001", "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "group_id"]
        self.assertEqual(len(failures), 1)
        self.assertIn("Missing", failures[0].reason)

    def test_empty_group_id(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "  ", "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "group_id"]
        self.assertEqual(len(failures), 1)
        self.assertIn("empty or whitespace", failures[0].reason)

    def test_non_string_group_id(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": 999, "x1": 0, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "group_id"]
        self.assertEqual(len(failures), 1)
        self.assertIn("must be a string", failures[0].reason)


class TestNumericScalarValidation(unittest.TestCase):
    """Test schema enforcement on 'x1' and 'x2' numeric fields."""

    def test_malformed_numeric_string_x1(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": "not_a_number", "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "x1"]
        self.assertEqual(len(failures), 1)
        self.assertIn("numeric scalar", failures[0].reason)

    def test_malformed_numeric_string_x2(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": "1.0", "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "x2"]
        self.assertEqual(len(failures), 1)
        self.assertIn("numeric scalar", failures[0].reason)

    def test_sequence_tensor_rejected_for_x1(self) -> None:
        test_containers = [[0, 1], (0, 1), {"a": 1}, {0, 1}]
        for container in test_containers:
            bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": container, "x2": 1, "label": 0}
            result = validate_dataset([bad_record])

            self.assertFalse(result.all_valid)
            failures = [f for f in result.failures if f.field == "x1"]
            self.assertEqual(len(failures), 1)
            self.assertIn("sequence/tensor", failures[0].reason)

    def test_sequence_tensor_rejected_for_x2(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": [1, 2, 3], "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "x2"]
        self.assertEqual(len(failures), 1)
        self.assertIn("sequence/tensor", failures[0].reason)

    def test_boolean_rejected_as_numeric(self) -> None:
        # In Python bool subclasses int; ensure True/False are strictly rejected
        bad_x1 = {"record_id": "rec_001", "group_id": "grp_01", "x1": True, "x2": 1, "label": 0}
        result_x1 = validate_dataset([bad_x1])
        self.assertFalse(result_x1.all_valid)
        failures_x1 = [f for f in result_x1.failures if f.field == "x1"]
        self.assertEqual(len(failures_x1), 1)
        self.assertIn("boolean", failures_x1[0].reason)

        bad_x2 = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": False, "label": 0}
        result_x2 = validate_dataset([bad_x2])
        self.assertFalse(result_x2.all_valid)
        failures_x2 = [f for f in result_x2.failures if f.field == "x2"]
        self.assertEqual(len(failures_x2), 1)
        self.assertIn("boolean", failures_x2[0].reason)

    def test_nan_and_inf_rejected(self) -> None:
        bad_record = {
            "record_id": "rec_001",
            "group_id": "grp_01",
            "x1": float("nan"),
            "x2": float("inf"),
            "label": 0,
        }
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        fields = {f.field for f in result.failures}
        self.assertIn("x1", fields)
        self.assertIn("x2", fields)

    def test_none_numeric_rejected(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": None, "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "x1"]
        self.assertEqual(len(failures), 1)
        self.assertIn("cannot be None", failures[0].reason)


class TestLabelValidation(unittest.TestCase):
    """Test schema enforcement on 'label' (binary integer: 0 or 1)."""

    def test_missing_label(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "label"]
        self.assertEqual(len(failures), 1)
        self.assertIn("Missing", failures[0].reason)

    def test_boolean_rejected_for_label(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": True}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "label"]
        self.assertEqual(len(failures), 1)
        self.assertIn("boolean", failures[0].reason)

    def test_float_label_rejected(self) -> None:
        bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": 1.0}
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        failures = [f for f in result.failures if f.field == "label"]
        self.assertEqual(len(failures), 1)
        self.assertIn("integer", failures[0].reason)

    def test_out_of_range_integer_label(self) -> None:
        invalid_labels = [2, -1, 100]
        for val in invalid_labels:
            bad_record = {"record_id": "rec_001", "group_id": "grp_01", "x1": 0, "x2": 1, "label": val}
            result = validate_dataset([bad_record])

            self.assertFalse(result.all_valid)
            failures = [f for f in result.failures if f.field == "label"]
            self.assertEqual(len(failures), 1)
            self.assertIn("binary (0 or 1)", failures[0].reason)


class TestMultipleFailuresAndReporting(unittest.TestCase):
    """Test that multiple failures in a single record are all reported without silent dropping."""

    def test_multiple_failures_in_single_record_reported(self) -> None:
        # This record has 4 failures: missing record_id, malformed x1, tensor x2, invalid label
        bad_record = {
            "group_id": "grp_01",
            "x1": "invalid_number",
            "x2": [1, 2, 3],
            "label": 99,
        }
        result = validate_dataset([bad_record])

        self.assertFalse(result.all_valid)
        self.assertEqual(len(result.failures), 4)

        failing_fields = {f.field for f in result.failures}
        self.assertEqual(failing_fields, {"record_id", "x1", "x2", "label"})

        # Record counts should accurately reflect 1 total, 1 invalid, 0 valid
        self.assertEqual(result.total_records, 1)
        self.assertEqual(result.invalid_count, 1)
        self.assertEqual(result.valid_count, 0)

    def test_failure_fields_populated_cleanly(self) -> None:
        bad_record = {"record_id": "rec_test_fail", "group_id": "grp_01", "x1": "bad", "x2": 1, "label": 0}
        result = validate_dataset([bad_record])

        self.assertEqual(len(result.failures), 1)
        failure = result.failures[0]

        self.assertEqual(failure.record_id, "rec_test_fail")
        self.assertEqual(failure.field, "x1")
        self.assertIsInstance(failure.reason, str)
        self.assertGreater(len(failure.reason.strip()), 0)

    def test_mixed_valid_and_invalid_records_not_dropped(self) -> None:
        records = [
            {"record_id": "rec_ok_1", "group_id": "grp_01", "x1": 10, "x2": 1, "label": 0},
            {"record_id": "rec_bad_1", "group_id": "grp_01", "x1": "bad", "x2": 1, "label": 0},
            {"record_id": "rec_ok_2", "group_id": "grp_02", "x1": 20, "x2": 1, "label": 1},
        ]
        result = validate_dataset(records)

        self.assertFalse(result.all_valid)
        self.assertEqual(result.total_records, 3)
        self.assertEqual(result.valid_count, 2)
        self.assertEqual(result.invalid_count, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].record_id, "rec_bad_1")

    def test_non_dict_record_handled_safely(self) -> None:
        records = ["not_a_dict_record"]
        result = validate_dataset(records)

        self.assertFalse(result.all_valid)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].field, "<record>")


if __name__ == "__main__":
    unittest.main()
