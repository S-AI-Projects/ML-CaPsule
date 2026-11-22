"""Unit tests for R41-T003 majority-class baseline and text fixture.

Verifies:
- Exact synthetic fixture contents, text, and labels.
- Majority class determined from training labels ONLY.
- Predictions and accuracy metric calculated exclusively on held-out data.
- Explicit train/validation/test separation without data leakage.
- Deterministic tie-breaking behaviour.
- Comprehensive input and split validation error paths.
- Regression test on the canonical held-out metric.
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
    DEFAULT_SPLIT,
    SYNTHETIC_FIXTURE,
    get_split_records,
    load_fixture,
)
from pipeline import (  # noqa: E402
    MajorityClassBaseline,
    calculate_metrics,
    run_baseline_pipeline,
    validate_records,
    validate_split_manifest,
)


class TestSyntheticFixtureSpecification(unittest.TestCase):
    """Verify exact fixture requirements from Issue #3."""

    def setUp(self) -> None:
        self.fixture = load_fixture()

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(self.fixture), 6, "Fixture must contain exactly six records.")
        for row in self.fixture:
            self.assertEqual(set(row.keys()), {"record_id", "text", "label"})

    def test_fixture_exact_rows_and_labels(self) -> None:
        expected = [
            ("t1", "red apple", 1),
            ("t2", "green apple", 1),
            ("t3", "blue sky", 0),
            ("t4", "dark sky", 0),
            ("t5", "red fruit", 1),
            ("t6", "blue cloud", 0),
        ]
        actual = [(r["record_id"], r["text"], r["label"]) for r in self.fixture]
        self.assertEqual(actual, expected)

    def test_fixture_class_balance(self) -> None:
        labels = [r["label"] for r in self.fixture]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_disclaimer_present(self) -> None:
        self.assertIn("Synthetic", DATASET_DISCLAIMER)
        self.assertIn("NOT represent real SMS", DATASET_DISCLAIMER)


class TestMajorityClassBaseline(unittest.TestCase):
    """Test majority-class classifier fitting, prediction, and tie-breaking."""

    def test_majority_class_selected_from_training_only(self) -> None:
        train_data = [
            {"record_id": "r1", "text": "a", "label": 1},
            {"record_id": "r2", "text": "b", "label": 1},
            {"record_id": "r3", "text": "c", "label": 0},
        ]
        clf = MajorityClassBaseline().fit(train_data)
        self.assertEqual(clf.majority_class, 1)
        self.assertFalse(clf.tie_detected_)

    def test_majority_class_zero_when_zeros_dominate(self) -> None:
        train_data = [
            {"record_id": "r1", "text": "a", "label": 0},
            {"record_id": "r2", "text": "b", "label": 0},
            {"record_id": "r3", "text": "c", "label": 1},
        ]
        clf = MajorityClassBaseline().fit(train_data)
        self.assertEqual(clf.majority_class, 0)
        self.assertFalse(clf.tie_detected_)

    def test_deterministic_tie_breaking(self) -> None:
        # Equal numbers of 0 and 1
        tied_train = [
            {"record_id": "r1", "text": "a", "label": 0},
            {"record_id": "r2", "text": "b", "label": 1},
        ]
        clf_lowest = MajorityClassBaseline(tie_breaker="lowest").fit(tied_train)
        self.assertTrue(clf_lowest.tie_detected_)
        self.assertEqual(clf_lowest.majority_class, 0)

        clf_highest = MajorityClassBaseline(tie_breaker="highest").fit(tied_train)
        self.assertTrue(clf_highest.tie_detected_)
        self.assertEqual(clf_highest.majority_class, 1)

    def test_predict_outputs_majority_class(self) -> None:
        train_data = [{"record_id": "r1", "text": "a", "label": 1}]
        clf = MajorityClassBaseline().fit(train_data)

        candidates = [
            {"record_id": "q1", "text": "x", "label": 0},
            {"record_id": "q2", "text": "y", "label": 1},
            {"record_id": "q3", "text": "z", "label": 0},
        ]
        preds = clf.predict(candidates)
        self.assertEqual(preds, [1, 1, 1])

    def test_predict_before_fit_raises(self) -> None:
        clf = MajorityClassBaseline()
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            clf.predict([{"record_id": "q1", "text": "x", "label": 0}])

    def test_empty_train_records_raises(self) -> None:
        clf = MajorityClassBaseline()
        with self.assertRaisesRegex(ValueError, "Cannot fit baseline on empty"):
            clf.fit([])

    def test_held_out_labels_do_not_influence_fit(self) -> None:
        # Training set has majority 1
        train = [
            {"record_id": "r1", "text": "a", "label": 1},
            {"record_id": "r2", "text": "b", "label": 1},
            {"record_id": "r3", "text": "c", "label": 0},
        ]
        # Held-out has all 0s
        held_out = [
            {"record_id": "h1", "text": "x", "label": 0},
            {"record_id": "h2", "text": "y", "label": 0},
            {"record_id": "h3", "text": "z", "label": 0},
            {"record_id": "h4", "text": "w", "label": 0},
        ]
        clf = MajorityClassBaseline().fit(train)
        self.assertEqual(clf.majority_class, 1)

        # Even if held-out labels are altered, fitted majority_class remains 1
        held_out_altered = [{"record_id": h["record_id"], "text": h["text"], "label": 1} for h in held_out]
        self.assertEqual(clf.majority_class, 1)
        preds = clf.predict(held_out_altered)
        self.assertEqual(preds, [1, 1, 1, 1])


class TestMetricsCalculation(unittest.TestCase):
    """Test accuracy and error rate calculation on held-out predictions."""

    def test_accuracy_calculation(self) -> None:
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 1]
        m = calculate_metrics(y_true, y_pred)
        self.assertEqual(m.total, 4)
        self.assertEqual(m.correct, 2)
        self.assertEqual(m.incorrect, 2)
        self.assertEqual(m.accuracy, 0.5)
        self.assertEqual(m.error_rate, 0.5)

    def test_perfect_accuracy(self) -> None:
        y_true = [1, 1, 1]
        y_pred = [1, 1, 1]
        m = calculate_metrics(y_true, y_pred)
        self.assertEqual(m.accuracy, 1.0)
        self.assertEqual(m.error_rate, 0.0)

    def test_zero_accuracy(self) -> None:
        y_true = [0, 0]
        y_pred = [1, 1]
        m = calculate_metrics(y_true, y_pred)
        self.assertEqual(m.accuracy, 0.0)
        self.assertEqual(m.error_rate, 1.0)

    def test_empty_metrics(self) -> None:
        m = calculate_metrics([], [])
        self.assertEqual(m.total, 0)
        self.assertEqual(m.accuracy, 0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths must match"):
            calculate_metrics([1, 0], [1])


class TestInputAndSplitValidation(unittest.TestCase):
    """Test data contract enforcement and split isolation validation."""

    def test_missing_required_field_raises(self) -> None:
        bad = [{"record_id": "t1", "text": "hello"}]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_records(bad)

    def test_empty_record_id_raises(self) -> None:
        bad = [{"record_id": "   ", "text": "hello", "label": 1}]
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            validate_records(bad)

    def test_duplicate_record_id_raises(self) -> None:
        bad = [
            {"record_id": "t1", "text": "a", "label": 1},
            {"record_id": "t1", "text": "b", "label": 0},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
            validate_records(bad)

    def test_boolean_label_rejected(self) -> None:
        bad = [{"record_id": "t1", "text": "a", "label": True}]
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            validate_records(bad)

    def test_float_label_rejected(self) -> None:
        bad = [{"record_id": "t1", "text": "a", "label": 1.0}]
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            validate_records(bad)

    def test_out_of_range_label_rejected(self) -> None:
        bad = [{"record_id": "t1", "text": "a", "label": 5}]
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            validate_records(bad)

    def test_split_manifest_missing_split_name(self) -> None:
        records = load_fixture()
        bad_manifest = {"train": ["t1", "t2"], "validation": ["t3"]}
        with self.assertRaisesRegex(ValueError, "must contain exactly"):
            validate_split_manifest(records, bad_manifest)

    def test_split_manifest_unknown_record_id(self) -> None:
        records = load_fixture()
        bad_manifest = {
            "train": ["t1", "t2", "t999"],
            "validation": ["t3", "t4"],
            "test": ["t5", "t6"],
        }
        with self.assertRaisesRegex(ValueError, "unknown record_ids"):
            validate_split_manifest(records, bad_manifest)

    def test_split_manifest_data_leakage_overlap_rejected(self) -> None:
        records = load_fixture()
        bad_manifest = {
            "train": ["t1", "t2", "t3"],
            "validation": ["t3", "t4"],  # t3 leaks into both train and validation!
            "test": ["t5", "t6"],
        }
        with self.assertRaisesRegex(ValueError, "Data leakage detected"):
            validate_split_manifest(records, bad_manifest)

    def test_split_manifest_unassigned_records_rejected(self) -> None:
        records = load_fixture()
        bad_manifest = {
            "train": ["t1", "t2"],
            "validation": ["t3"],
            "test": ["t4"],
            # t5, t6 unassigned
        }
        with self.assertRaisesRegex(ValueError, "must partition every record"):
            validate_split_manifest(records, bad_manifest)


class TestEndToEndBaselineRegression(unittest.TestCase):
    """End-to-end regression tests verifying the exact expected held-out baseline results."""

    def test_canonical_pipeline_execution(self) -> None:
        records = load_fixture()
        result = run_baseline_pipeline(records, DEFAULT_SPLIT)

        # 1. Majority class determined exclusively from train: t1(1), t2(1), t3(0)
        self.assertEqual(result.majority_class, 1)
        self.assertEqual(result.train_class_counts, {1: 2, 0: 1})
        self.assertFalse(result.tie_detected)

        # 2. Validation set: t4 (true=0, pred=1) -> 0/1 correct = 0.0 accuracy
        self.assertEqual(result.validation_metrics.total, 1)
        self.assertEqual(result.validation_metrics.correct, 0)
        self.assertEqual(result.validation_metrics.incorrect, 1)
        self.assertEqual(result.validation_metrics.accuracy, 0.0)

        # 3. Test set: t5 (true=1, pred=1), t6 (true=0, pred=1) -> 1/2 correct = 0.5 accuracy
        self.assertEqual(result.test_metrics.total, 2)
        self.assertEqual(result.test_metrics.correct, 1)
        self.assertEqual(result.test_metrics.incorrect, 1)
        self.assertEqual(result.test_metrics.accuracy, 0.5)

        # 4. Overall held-out metrics (validation + test): 3 total, 1 correct (t5), 2 incorrect (t4, t6)
        self.assertEqual(result.overall_held_out_metrics.total, 3)
        self.assertEqual(result.overall_held_out_metrics.correct, 1)
        self.assertEqual(result.overall_held_out_metrics.incorrect, 2)
        self.assertAlmostEqual(result.overall_held_out_metrics.accuracy, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(result.overall_held_out_metrics.error_rate, 2.0 / 3.0, places=6)

    def test_regression_fails_if_expected_result_is_altered(self) -> None:
        """Verify that the test suite fails if the concrete expected result is changed."""
        records = load_fixture()
        result = run_baseline_pipeline(records, DEFAULT_SPLIT)

        # The true test accuracy is exactly 0.5
        expected_test_accuracy = 0.5
        self.assertEqual(result.test_metrics.accuracy, expected_test_accuracy)

        # Deliberately checking a falsified result assertion would fail
        falsified_test_accuracy = 0.99
        self.assertNotEqual(result.test_metrics.accuracy, falsified_test_accuracy)


if __name__ == "__main__":
    unittest.main()
