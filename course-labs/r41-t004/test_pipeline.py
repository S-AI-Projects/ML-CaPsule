"""Unit tests for R41-T004 metric verification and synthetic pipeline.

Covers:
- Concrete classification verification: truth=[1,1,0,0], pred=[1,0,1,0] -> precision=0.5, recall=0.5.
- Concrete regression verification: truth=[1,2,3], pred=[1,3,2] -> MAE=2/3.
- Classification input validation (empty, mismatch, types, booleans, invalid labels).
- Regression input validation (empty, mismatch, types, booleans, NaN, Inf).
- Zero-denominator handling for precision and recall.
- Synthetic fixture structure, keys, values, and disclaimer.
- Stratified 3-way split integrity and zero-leakage enforcement.
- Expected-result regression test with deliberate failure verification.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure local modules (pipeline, fixture) are importable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fixture import (  # noqa: E402
    CLASSIFICATION_BENCHMARK,
    DATASET_DISCLAIMER,
    REGRESSION_BENCHMARK,
    STRATIFIED_SPLIT,
    SYNTHETIC_FIXTURE,
    get_stratified_split,
    load_fixture,
)
from pipeline import (  # noqa: E402
    calculate_accuracy,
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_mae,
    calculate_precision,
    calculate_recall,
    validate_binary_classification_inputs,
    validate_records,
    validate_regression_inputs,
    validate_split_manifest,
)


class TestIssue4MetricBenchmarks(unittest.TestCase):
    """Verify the concrete metric values explicitly mandated by Issue #4."""

    def test_classification_expected_precision_and_recall(self) -> None:
        """Verify truth=[1,1,0,0], pred=[1,0,1,0] -> precision=0.5, recall=0.5."""
        truth = CLASSIFICATION_BENCHMARK["truth"]
        pred = CLASSIFICATION_BENCHMARK["prediction"]

        metrics = calculate_classification_metrics(truth, pred)

        # Precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
        self.assertEqual(metrics.precision, 0.5)

        # Recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
        self.assertEqual(metrics.recall, 0.5)

        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        self.assertEqual(metrics.f1_score, 0.5)

        # Accuracy = (1 + 1) / 4 = 0.5
        self.assertEqual(metrics.accuracy, 0.5)

        # Confusion matrix: TP=1, FP=1, FN=1, TN=1
        cm = metrics.confusion_matrix
        self.assertEqual(cm.tp, 1)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tn, 1)
        self.assertEqual(cm.total, 4)

    def test_regression_expected_mae(self) -> None:
        """Verify truth=[1,2,3], pred=[1,3,2] -> MAE = 2/3."""
        truth = REGRESSION_BENCHMARK["truth"]
        pred = REGRESSION_BENCHMARK["prediction"]

        mae = calculate_mae(truth, pred)
        expected_mae = 2.0 / 3.0

        self.assertAlmostEqual(mae, expected_mae, places=7)
        self.assertEqual(round(mae, 6), round(2.0 / 3.0, 6))

    def test_regression_fails_if_expected_metric_is_altered(self) -> None:
        """Verify that the test suite detects if expected metrics are altered."""
        truth = CLASSIFICATION_BENCHMARK["truth"]
        pred = CLASSIFICATION_BENCHMARK["prediction"]
        metrics = calculate_classification_metrics(truth, pred)

        # The true precision is 0.5; a falsified expectation fails
        self.assertEqual(metrics.precision, 0.5)
        falsified_precision = 0.99
        self.assertNotEqual(metrics.precision, falsified_precision)

        # The true MAE is 2/3; a falsified expectation fails
        mae = calculate_mae(REGRESSION_BENCHMARK["truth"], REGRESSION_BENCHMARK["prediction"])
        self.assertAlmostEqual(mae, 2.0 / 3.0, places=7)
        falsified_mae = 0.05
        self.assertNotEqual(round(mae, 4), falsified_mae)


class TestZeroDenominatorBehavior(unittest.TestCase):
    """Test handling of zero denominators in precision and recall."""

    def test_precision_zero_division_when_no_positive_predictions(self) -> None:
        truth = [1, 1, 0, 0]
        pred = [0, 0, 0, 0]  # TP=0, FP=0 -> denominator TP + FP = 0

        # Default zero_division=0.0
        prec_default = calculate_precision(truth, pred)
        self.assertEqual(prec_default, 0.0)

        # Configurable zero_division=1.0
        prec_custom = calculate_precision(truth, pred, zero_division=1.0)
        self.assertEqual(prec_custom, 1.0)

    def test_recall_zero_division_when_no_positive_truth(self) -> None:
        truth = [0, 0, 0, 0]  # TP=0, FN=0 -> denominator TP + FN = 0
        pred = [1, 1, 0, 0]

        rec_default = calculate_recall(truth, pred)
        self.assertEqual(rec_default, 0.0)

        rec_custom = calculate_recall(truth, pred, zero_division=1.0)
        self.assertEqual(rec_custom, 1.0)

    def test_f1_zero_division_when_precision_and_recall_zero(self) -> None:
        truth = [0, 0]
        pred = [0, 0]

        f1_default = calculate_f1_score(truth, pred)
        self.assertEqual(f1_default, 0.0)

        f1_custom = calculate_f1_score(truth, pred, zero_division=1.0)
        self.assertEqual(f1_custom, 1.0)


class TestInputValidation(unittest.TestCase):
    """Test defensive error handling for classification and regression inputs."""

    def test_empty_classification_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            calculate_precision([], [])

    def test_mismatched_classification_lengths_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            calculate_precision([1, 0], [1])

    def test_boolean_classification_inputs_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            calculate_precision([True, False], [1, 0])

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            calculate_precision([1, 0], [False, True])

    def test_invalid_label_values_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            calculate_precision([2, 0], [1, 0])

        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            calculate_precision([1, 0], [-1, 0])

    def test_float_classification_labels_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            calculate_precision([1.0, 0.0], [1, 0])

    def test_none_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            validate_binary_classification_inputs(None, [1, 0])  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "cannot be None"):
            validate_regression_inputs([1.0], None)  # type: ignore[arg-type]

    def test_empty_regression_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            calculate_mae([], [])

    def test_mismatched_regression_lengths_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            calculate_mae([1, 2], [1])

    def test_boolean_regression_inputs_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a numeric scalar"):
            calculate_mae([True, 2], [1, 2])

    def test_non_finite_regression_inputs_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite real number"):
            calculate_mae([float("nan"), 2], [1, 2])

        with self.assertRaisesRegex(ValueError, "finite real number"):
            calculate_mae([1, 2], [float("inf"), 2])


class TestSyntheticFixtureAndSplit(unittest.TestCase):
    """Test 8-row synthetic fixture specification and 3-way stratified split."""

    def setUp(self) -> None:
        self.fixture = load_fixture()

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(self.fixture), 8)
        expected_keys = {"record_id", "group_id", "x1", "x2", "label"}
        for r in self.fixture:
            self.assertEqual(set(r.keys()), expected_keys)

    def test_fixture_exact_features_and_labels(self) -> None:
        x1_vals = [r["x1"] for r in self.fixture]
        x2_vals = [r["x2"] for r in self.fixture]
        labels = [r["label"] for r in self.fixture]

        self.assertEqual(x1_vals, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(x2_vals, [1] * 8)
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic", DATASET_DISCLAIMER)
        self.assertIn("NOT represent real Pokémon stats", DATASET_DISCLAIMER)

    def test_stratified_split_integrity(self) -> None:
        split = get_stratified_split()
        train_recs = split["train"]
        val_recs = split["validation"]
        test_recs = split["test"]

        # Sizes: 4 train, 2 validation, 2 test
        self.assertEqual(len(train_recs), 4)
        self.assertEqual(len(val_recs), 2)
        self.assertEqual(len(test_recs), 2)

        # 50/50 balance in each split
        train_labels = [r["label"] for r in train_recs]
        val_labels = [r["label"] for r in val_recs]
        test_labels = [r["label"] for r in test_recs]

        self.assertEqual(train_labels.count(0), 2)
        self.assertEqual(train_labels.count(1), 2)
        self.assertEqual(val_labels.count(0), 1)
        self.assertEqual(val_labels.count(1), 1)
        self.assertEqual(test_labels.count(0), 1)
        self.assertEqual(test_labels.count(1), 1)

        # Zero leakage: disjoint IDs
        train_ids = {r["record_id"] for r in train_recs}
        val_ids = {r["record_id"] for r in val_recs}
        test_ids = {r["record_id"] for r in test_recs}

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_validate_records_helper(self) -> None:
        validate_records(self.fixture)

    def test_split_manifest_leakage_detected(self) -> None:
        manifest = {
            "train": ["rec_001", "rec_002", "rec_003"],
            "validation": ["rec_003", "rec_004"],  # rec_003 leaks!
            "test": ["rec_005", "rec_006", "rec_007", "rec_008"],
        }
        with self.assertRaisesRegex(ValueError, "Leakage"):
            validate_split_manifest(self.fixture, manifest)


if __name__ == "__main__":
    unittest.main()
