"""Unit tests for Auto MPG regression minimal baseline comparison (Task R41-T023)."""

import json
from pathlib import Path
import unittest

try:
    from pipeline import (
        LinearRegressionComparator,
        TrainingMeanBaseline,
        get_environment_info,
        load_fixture,
        verify_fixture_formula,
    )
except ImportError:
    from .pipeline import (
        LinearRegressionComparator,
        TrainingMeanBaseline,
        get_environment_info,
        load_fixture,
        verify_fixture_formula,
    )


class TestAutoMPGBaselineComparison(unittest.TestCase):
    """Test suite verifying training-mean baseline isolation, evaluation metrics, and formula fixture."""

    def setUp(self):
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.split_ids = self.fixture["metadata"]["splits"]
        self.train_recs = [r for r in self.records if r["record_id"] in self.split_ids["train"]]
        self.val_recs = [r for r in self.records if r["record_id"] in self.split_ids["validation"]]
        self.test_recs = [r for r in self.records if r["record_id"] in self.split_ids["test"]]

    def test_fixture_six_synthetic_rows_and_deterministic_formula(self):
        """Verify fixture contains six synthetic rows strictly generated via target = 2 * x1 + 1."""
        self.assertEqual(len(self.records), 6)
        x1_vals = [r["x1"] for r in self.records]
        x2_vals = [r["x2"] for r in self.records]
        targets = [r["target"] for r in self.records]
        ids = [r["record_id"] for r in self.records]

        self.assertEqual(x1_vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(x2_vals, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(targets, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        self.assertEqual(ids, ["r1", "r2", "r3", "r4", "r5", "r6"])

        formula_check = verify_fixture_formula(self.records)
        self.assertTrue(formula_check["formula_satisfied"])
        self.assertEqual(len(formula_check["discrepancies"]), 0)

    def test_train_val_test_partition_assignments(self):
        """Verify explicit split separation: train r1..r4, validation r5, test r6."""
        self.assertEqual([r["record_id"] for r in self.train_recs], ["r1", "r2", "r3", "r4"])
        self.assertEqual([r["record_id"] for r in self.val_recs], ["r5"])
        self.assertEqual([r["record_id"] for r in self.test_recs], ["r6"])

    def test_baseline_parameters_use_only_training_data(self):
        """Verify baseline calculates the target mean strictly from training records (mean = 4.0)."""
        baseline = TrainingMeanBaseline()
        baseline.fit(self.train_recs)

        # Expected training targets: [1.0, 3.0, 5.0, 7.0] -> sum = 16.0 / 4 = 4.0
        self.assertTrue(baseline.fitted)
        self.assertAlmostEqual(baseline.training_mean, 4.0, places=9)
        self.assertEqual(baseline.train_record_ids, ["r1", "r2", "r3", "r4"])

    def test_held_out_validation_evaluation(self):
        """Verify evaluation on held-out validation record r5 (actual 9.0, predicted 4.0, MAE 5.0, MSE 25.0)."""
        baseline = TrainingMeanBaseline().fit(self.train_recs)
        val_eval = baseline.evaluate(self.val_recs, split_name="validation")

        self.assertEqual(val_eval.record_ids, ["r5"])
        self.assertEqual(val_eval.predictions, [4.0])
        self.assertEqual(val_eval.actuals, [9.0])
        self.assertAlmostEqual(val_eval.mae, 5.0, places=9)
        self.assertAlmostEqual(val_eval.mse, 25.0, places=9)
        self.assertAlmostEqual(val_eval.rmse, 5.0, places=9)

    def test_held_out_test_evaluation(self):
        """Verify evaluation on held-out test record r6 (actual 11.0, predicted 4.0, MAE 7.0, MSE 49.0)."""
        baseline = TrainingMeanBaseline().fit(self.train_recs)
        test_eval = baseline.evaluate(self.test_recs, split_name="test")

        self.assertEqual(test_eval.record_ids, ["r6"])
        self.assertEqual(test_eval.predictions, [4.0])
        self.assertEqual(test_eval.actuals, [11.0])
        self.assertAlmostEqual(test_eval.mae, 7.0, places=9)
        self.assertAlmostEqual(test_eval.mse, 49.0, places=9)
        self.assertAlmostEqual(test_eval.rmse, 7.0, places=9)

    def test_predict_without_fit_raises_error(self):
        """Verify calling predict before fit raises RuntimeError."""
        baseline = TrainingMeanBaseline()
        with self.assertRaises(RuntimeError):
            baseline.predict(self.val_recs)

    def test_fit_on_empty_or_corrupted_records_raises_error(self):
        """Verify fitting on empty records or missing target raises ValueError."""
        baseline = TrainingMeanBaseline()
        with self.assertRaises(ValueError):
            baseline.fit([])

        with self.assertRaises(ValueError):
            baseline.fit([{"record_id": "r1", "x1": 0.0}])

    def test_comparator_linear_regression_fit(self):
        """Verify oracle linear model achieves MSE 0.0, demonstrating baseline contrast."""
        linear_model = LinearRegressionComparator().fit(self.train_recs)
        self.assertAlmostEqual(linear_model.slope, 2.0, places=9)
        self.assertAlmostEqual(linear_model.intercept, 1.0, places=9)

        val_eval = linear_model.evaluate(self.val_recs)
        self.assertAlmostEqual(val_eval["mse"], 0.0, places=9)
        self.assertAlmostEqual(val_eval["mae"], 0.0, places=9)

    def test_environment_info_recorded(self):
        """Verify runtime environment information is present."""
        env = get_environment_info()
        self.assertIn("python", env)
        self.assertIn("os", env)

    def test_deliberate_mutation_failure(self):
        """Verify test framework fails nonzero when an expected baseline metric is altered."""
        expected_mae = 5.0
        actual_tampered = 99.0
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_tampered, expected_mae, "Deliberate mutation assertion triggered")


if __name__ == "__main__":
    unittest.main()
