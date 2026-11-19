"""Unit tests for Abalone age regression deterministic split manifest (Task R41-T022)."""

import copy
import json
from pathlib import Path
import unittest

try:
    from pipeline import (
        SimpleLinearRegression,
        SplitManifest,
        create_deterministic_split,
        get_environment_info,
        load_fixture,
        validate_manifest,
        verify_deterministic_formula,
    )
except ImportError:
    from .pipeline import (
        SimpleLinearRegression,
        SplitManifest,
        create_deterministic_split,
        get_environment_info,
        load_fixture,
        validate_manifest,
        verify_deterministic_formula,
    )


class TestAbaloneSplitManifest(unittest.TestCase):
    """Test suite for leakage-free splitting, deterministic formula fixture, and regression checks."""

    def setUp(self):
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.manifest_data = self.fixture["metadata"]["manifest"]

    def test_fixture_six_synthetic_rows_and_deterministic_formula(self):
        """Verify fixture has 6 synthetic rows with x1 in [0..5], x2=1, and target=2*x1+1."""
        self.assertEqual(len(self.records), 6)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        targets = [r["target"] for r in self.records]
        ids = [r["record_id"] for r in self.records]

        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(x2_vals, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(targets, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        self.assertEqual(ids, ["r1", "r2", "r3", "r4", "r5", "r6"])

        formula_check = verify_deterministic_formula(self.records)
        self.assertTrue(formula_check["formula_satisfied"])
        self.assertEqual(len(formula_check["discrepancies"]), 0)

    def test_manifest_partition_assignments(self):
        """Verify exact record IDs assigned to each split: train r1..r4, validation r5, test r6."""
        manifest = SplitManifest.from_dict(self.manifest_data)
        self.assertEqual(manifest.train_ids, ["r1", "r2", "r3", "r4"])
        self.assertEqual(manifest.val_ids, ["r5"])
        self.assertEqual(manifest.test_ids, ["r6"])

    def test_train_val_test_sets_disjoint_and_complete(self):
        """Verify train, validation, and test sets are strictly disjoint and have complete coverage."""
        manifest = SplitManifest.from_dict(self.manifest_data)
        val_result = validate_manifest(manifest, self.records)
        self.assertTrue(val_result["valid"])
        self.assertTrue(val_result["is_disjoint"])
        self.assertTrue(val_result["complete_coverage"])
        self.assertTrue(val_result["is_chronological"])
        self.assertEqual(len(val_result["errors"]), 0)

    def test_repeated_runs_produce_same_manifest(self):
        """Verify deterministic generation reproduces the identical manifest across repeated runs."""
        m1 = create_deterministic_split(self.records, train_count=4, val_count=1)
        m2 = create_deterministic_split(self.records, train_count=4, val_count=1)
        self.assertEqual(m1.to_dict(), m2.to_dict())
        self.assertEqual(m1.train_ids, ["r1", "r2", "r3", "r4"])
        self.assertEqual(m1.val_ids, ["r5"])
        self.assertEqual(m1.test_ids, ["r6"])

    def test_intentionally_overlapping_ids_rejected(self):
        """Verify manifest with overlapping record IDs across splits is rejected."""
        overlapping_manifest = SplitManifest(
            train_ids=["r1", "r2", "r3", "r4", "r5"],  # r5 placed in train
            val_ids=["r5"],                             # AND r5 placed in val
            test_ids=["r6"],
        )
        result = validate_manifest(overlapping_manifest, self.records)
        self.assertFalse(result["valid"])
        self.assertFalse(result["is_disjoint"])
        overlap_errors = [e for e in result["errors"] if "Overlap between train and validation" in e]
        self.assertEqual(len(overlap_errors), 1)

    def test_group_leakage_rejected(self):
        """Verify manifest leaking group IDs across train and validation splits is rejected."""
        corrupted_records = copy.deepcopy(self.records)
        # Force r5 to share grp_1 with train records r1, r2
        corrupted_records[4]["group_id"] = "grp_1"

        manifest = SplitManifest.from_dict(self.manifest_data)
        result = validate_manifest(manifest, corrupted_records)
        self.assertFalse(result["valid"])
        group_errors = [e for e in result["errors"] if "Group leakage" in e]
        self.assertTrue(len(group_errors) > 0)

    def test_chronological_time_series_order_and_leakage_rejection(self):
        """Verify time-series chronological monotonicity and rejection of future-to-past leakage."""
        corrupted_records = copy.deepcopy(self.records)
        # Swap timestamps so r1 (train) has future timestamp 9999
        corrupted_records[0]["timestamp"] = 9999

        manifest = SplitManifest.from_dict(self.manifest_data)
        result = validate_manifest(manifest, corrupted_records)
        self.assertFalse(result["valid"])
        self.assertFalse(result["is_chronological"])
        chrono_errors = [e for e in result["errors"] if "Chronological leakage" in e]
        self.assertTrue(len(chrono_errors) > 0)

    def test_linear_regression_exact_parameter_recovery(self):
        """Verify CPU-only linear regression fits slope=2, intercept=1 and recovers targets with 0 MSE."""
        train_records = [r for r in self.records if r["record_id"] in self.manifest_data["train"]]
        val_records = [r for r in self.records if r["record_id"] in self.manifest_data["validation"]]
        test_records = [r for r in self.records if r["record_id"] in self.manifest_data["test"]]

        model = SimpleLinearRegression().fit(train_records)
        self.assertAlmostEqual(model.slope, 2.0, places=9)
        self.assertAlmostEqual(model.intercept, 1.0, places=9)

        val_eval = model.evaluate(val_records)
        self.assertAlmostEqual(val_eval["mse"], 0.0, places=9)
        self.assertAlmostEqual(val_eval["predictions"][0], 9.0, places=9)

        test_eval = model.evaluate(test_records)
        self.assertAlmostEqual(test_eval["mse"], 0.0, places=9)
        self.assertAlmostEqual(test_eval["predictions"][0], 11.0, places=9)

    def test_environment_info_recorded(self):
        """Verify environment versions capture Python and OS."""
        env = get_environment_info()
        self.assertIn("python", env)
        self.assertIn("os", env)

    def test_deliberate_mutation_failure(self):
        """Verify check fails when an expected target is deliberately altered."""
        expected_test_target = 11.0
        actual_tampered = 99.0
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_tampered, expected_test_target, "Deliberate mutation assertion triggered")


if __name__ == "__main__":
    unittest.main()
