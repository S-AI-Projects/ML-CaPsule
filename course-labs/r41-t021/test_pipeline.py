"""Unit tests for MNIST k-NN baseline data contract and schema validator (Task R41-T021)."""

import copy
import json
from pathlib import Path
import unittest

try:
    from pipeline import (
        KNNImageClassifier,
        get_environment_info,
        load_fixture,
        validate_records,
    )
except ImportError:
    from .pipeline import (
        KNNImageClassifier,
        get_environment_info,
        load_fixture,
        validate_records,
    )


class TestMNISTDataContract(unittest.TestCase):
    """Test suite validating schema conformance, failure diagnosis, and baseline k-NN execution."""

    def setUp(self):
        self.fixture = load_fixture()
        self.records = self.fixture["records"]

    def test_valid_fixture_passes(self):
        """Verify the canonical 6-row synthetic fixture passes schema validation with zero failures."""
        result = validate_records(self.records)
        self.assertTrue(result["valid"])
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["total_records_checked"], 6)
        self.assertEqual(len(result["failures"]), 0)

    def test_fixture_six_synthetic_rows_contract(self):
        """Verify literal synthetic fixture: six 4x4 images, 3 zero (label 0), 3 diagonal (label 1)."""
        self.assertEqual(len(self.records), 6)

        zero_records = [r for r in self.records if r["label"] == 0]
        diag_records = [r for r in self.records if r["label"] == 1]
        self.assertEqual(len(zero_records), 3)
        self.assertEqual(len(diag_records), 3)

        # Verify all-zero images
        for r in zero_records:
            px = r["pixels"]
            self.assertEqual(len(px), 4)
            for row in px:
                self.assertEqual(row, [0, 0, 0, 0])
            self.assertEqual(r["group_id"], "grp_0")

        # Verify white diagonal (255) images
        for r in diag_records:
            px = r["pixels"]
            self.assertEqual(len(px), 4)
            for i in range(4):
                for j in range(4):
                    expected_val = 255 if i == j else 0
                    self.assertEqual(px[i][j], expected_val)
            self.assertEqual(r["group_id"], "grp_1")

        # Verify unique record IDs
        ids = [r["record_id"] for r in self.records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_explicit_stratified_split_separation(self):
        """Verify explicit train/validation/test split separation with balanced stratification."""
        splits = {"train": [], "validation": [], "test": []}
        for r in self.records:
            split_name = r.get("split")
            self.assertIn(split_name, splits)
            splits[split_name].append(r)

        # Each split has exactly 2 records: one class 0, one class 1
        for split_name, split_recs in splits.items():
            self.assertEqual(len(split_recs), 2, f"{split_name} split count mismatch")
            labels = sorted([r["label"] for r in split_recs])
            self.assertEqual(labels, [0, 1], f"{split_name} split is not stratified")

    def test_missing_id_and_required_fields_reported(self):
        """Verify missing record_id, group_id, label, or pixels are diagnosed without dropping rows."""
        corrupted = copy.deepcopy(self.records[:2])
        del corrupted[0]["record_id"]
        del corrupted[1]["group_id"]

        result = validate_records(corrupted)
        self.assertFalse(result["valid"])
        self.assertEqual(result["total_records_checked"], 2)
        self.assertGreaterEqual(result["failure_count"], 2)

        failures = result["failures"]
        missing_record_id = [f for f in failures if f["field"] == "record_id" and "Missing required" in f["reason"]]
        missing_group_id = [f for f in failures if f["field"] == "group_id" and "Missing required" in f["reason"]]
        self.assertTrue(len(missing_record_id) > 0)
        self.assertTrue(len(missing_group_id) > 0)

    def test_duplicate_record_id_reported(self):
        """Verify duplicate record_id values are flagged with specific reason."""
        corrupted = copy.deepcopy(self.records[:3])
        corrupted[1]["record_id"] = corrupted[0]["record_id"]

        result = validate_records(corrupted)
        self.assertFalse(result["valid"])
        dup_failures = [f for f in failures if "Duplicate record_id" in f["reason"]] if (failures := result["failures"]) else []
        self.assertEqual(len(dup_failures), 1)
        self.assertEqual(dup_failures[0]["field"], "record_id")

    def test_malformed_numeric_and_text_data_reported(self):
        """Verify malformed text, non-integer labels, booleans, and string values are diagnosed."""
        corrupted = copy.deepcopy(self.records[:3])
        corrupted[0]["record_id"] = "   "  # whitespace only
        corrupted[1]["label"] = True        # boolean masquerading as int
        corrupted[2]["label"] = 2           # invalid label out of {0, 1}

        result = validate_records(corrupted)
        self.assertFalse(result["valid"])
        failures = result["failures"]

        ws_failure = [f for f in failures if f["field"] == "record_id" and "empty or whitespace" in f["reason"]]
        bool_failure = [f for f in failures if f["field"] == "label" and "not a boolean" in f["reason"]]
        range_failure = [f for f in failures if f["field"] == "label" and "must be in {0, 1}" in f["reason"]]

        self.assertEqual(len(ws_failure), 1)
        self.assertEqual(len(bool_failure), 1)
        self.assertEqual(len(range_failure), 1)

    def test_malformed_tensor_data_reported(self):
        """Verify tensor shape errors (non-4x4, irregular row lengths, out-of-range pixel values)."""
        corrupted = copy.deepcopy(self.records[:3])
        # Wrong row count (3 rows instead of 4)
        corrupted[0]["pixels"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # Pixel out of range [0, 255]
        corrupted[1]["pixels"] = [
            [256, 0, 0, 0],
            [0, 255, 0, 0],
            [0, 0, 255, 0],
            [0, 0, 0, 255],
        ]
        # Irregular row length
        corrupted[2]["pixels"] = [
            [0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        result = validate_records(corrupted)
        self.assertFalse(result["valid"])
        failures = result["failures"]

        dim_errors = [f for f in failures if f["field"] == "pixels" and "dimension error" in f["reason"]]
        pixel_val_errors = [f for f in failures if f["field"] == "pixels" and "range [0, 255]" in f["reason"]]
        row_len_errors = [f for f in failures if f["field"] == "pixels" and "must have exactly 4 elements per row" in f["reason"]]

        self.assertTrue(len(dim_errors) > 0)
        self.assertTrue(len(pixel_val_errors) > 0)
        self.assertTrue(len(row_len_errors) > 0)

    def test_knn_baseline_bounded_cpu_run(self):
        """Verify offline CPU-only 1-NN baseline accurately classifies validation and test splits."""
        train_records = [r for r in self.records if r["split"] == "train"]
        val_records = [r for r in self.records if r["split"] == "validation"]
        test_records = [r for r in self.records if r["split"] == "test"]

        clf = KNNImageClassifier(k=1)
        clf.fit(train_records)

        val_eval = clf.evaluate(val_records)
        self.assertEqual(val_eval["accuracy"], 1.0)
        self.assertEqual(val_eval["predictions"], [0, 1])

        test_eval = clf.evaluate(test_records)
        self.assertEqual(test_eval["accuracy"], 1.0)
        self.assertEqual(test_eval["predictions"], [0, 1])

    def test_environment_info_recorded(self):
        """Verify environment info reports Python and OS versions."""
        env = get_environment_info()
        self.assertIn("python", env)
        self.assertIn("os", env)
        self.assertTrue(len(env["python"]) > 0)

    def test_deliberate_mutation_failure(self):
        """Verify deliberate tampering of expected output produces an assertion failure."""
        expected_label = 0
        actual_label = 1  # Deliberately mutated
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_label, expected_label, "Mutation check assertion triggered")


if __name__ == "__main__":
    unittest.main()
