"""Unit tests for Task R41-T006: Forest fires – regression smoke test."""

import math
import unittest
from typing import Any, Dict, List

from fixture import (
    DISCLAIMER,
    FEATURE_ORDER,
    ID_FIELD,
    SPLIT_MANIFEST,
    SYNTHETIC_RECORDS,
    TARGET_FIELD,
    get_split_records,
)
from pipeline import (
    MAX_ALLOWED_EPOCHS,
    MAX_ALLOWED_SAMPLES,
    MAX_ALLOWED_STEPS,
    CPURegressionSmokeModel,
    ConstantBaselineModel,
    PipelineBoundaryError,
    SchemaValidationError,
    run_regression_smoke_pipeline,
)


class TestSyntheticFixture(unittest.TestCase):
    """Verify synthetic fixture properties and disclaimers."""

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic pipeline-verification fixture", DISCLAIMER)
        self.assertIn("Not real-world or Forest Fires regression data", DISCLAIMER)

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(SYNTHETIC_RECORDS), 6)
        expected_keys = {ID_FIELD, "x1", "x2", TARGET_FIELD}
        for rec in SYNTHETIC_RECORDS:
            self.assertEqual(set(rec.keys()), expected_keys)

    def test_deterministic_formula_and_features(self) -> None:
        expected_x1 = [0, 1, 2, 3, 4, 5]
        for idx, rec in enumerate(SYNTHETIC_RECORDS):
            self.assertEqual(rec["x1"], expected_x1[idx])
            self.assertEqual(rec["x2"], 1)
            self.assertEqual(rec["target"], 2 * rec["x1"] + 1)

    def test_split_manifest_disjoint(self) -> None:
        self.assertEqual(SPLIT_MANIFEST["train"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(SPLIT_MANIFEST["validation"], ["r5"])
        self.assertEqual(SPLIT_MANIFEST["test"], ["r6"])

        all_ids = SPLIT_MANIFEST["train"] + SPLIT_MANIFEST["validation"] + SPLIT_MANIFEST["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_ids), 6)


class TestBoundedCPUSmokeRun(unittest.TestCase):
    """Test CPU smoke run execution, bounds, shapes, and finite metrics."""

    def setUp(self) -> None:
        splits = get_split_records()
        self.train_records = splits["train"]
        self.val_records = splits["validation"]
        self.test_records = splits["test"]

    def test_smoke_pipeline_finite_outputs_and_loss(self) -> None:
        result = run_regression_smoke_pipeline(
            self.train_records, self.val_records, self.test_records, seed=42
        )
        self.assertTrue(result["all_outputs_finite"])
        self.assertFalse(math.isnan(result["train_loss_mse"]))
        self.assertFalse(math.isinf(result["train_loss_mse"]))
        self.assertFalse(math.isnan(result["val_mse"]))
        self.assertFalse(math.isnan(result["test_mse"]))

    def test_expected_prediction_shapes_and_step_count(self) -> None:
        result = run_regression_smoke_pipeline(
            self.train_records, self.val_records, self.test_records, seed=42
        )
        # 4 training records in 1 epoch -> 4 update steps <= 20
        self.assertEqual(result["step_count"], 4)
        self.assertLessEqual(result["step_count"], MAX_ALLOWED_STEPS)
        self.assertEqual(result["epoch_count"], 1)
        self.assertLessEqual(result["epoch_count"], MAX_ALLOWED_EPOCHS)

        # Expected shapes
        self.assertEqual(result["val_prediction_shape"], (1,))
        self.assertEqual(len(result["val_predictions"]), 1)
        self.assertEqual(result["test_prediction_shape"], (1,))
        self.assertEqual(len(result["test_predictions"]), 1)

    def test_fixed_seed_determinism(self) -> None:
        res1 = run_regression_smoke_pipeline(self.train_records, self.val_records, self.test_records, seed=42)
        res2 = run_regression_smoke_pipeline(self.train_records, self.val_records, self.test_records, seed=42)

        self.assertEqual(res1["train_loss_mse"], res2["train_loss_mse"])
        self.assertEqual(res1["val_predictions"], res2["val_predictions"])
        self.assertEqual(res1["test_predictions"], res2["test_predictions"])

    def test_sample_count_boundary_enforced(self) -> None:
        # Exceeding 32 samples must raise PipelineBoundaryError
        oversized = [
            {"record_id": f"rec_{i:03d}", "x1": i, "x2": 1, "target": 2 * i + 1}
            for i in range(MAX_ALLOWED_SAMPLES + 1)
        ]
        model = CPURegressionSmokeModel()
        with self.assertRaises(PipelineBoundaryError):
            model.fit(oversized)

    def test_epoch_and_step_bounds_enforced(self) -> None:
        with self.assertRaises(PipelineBoundaryError):
            CPURegressionSmokeModel(max_epochs=MAX_ALLOWED_EPOCHS + 1)
        with self.assertRaises(PipelineBoundaryError):
            CPURegressionSmokeModel(max_steps=MAX_ALLOWED_STEPS + 1)


class TestConstantBaselineModel(unittest.TestCase):
    """Test explicitly labelled naive baseline model."""

    def test_naive_baseline_properties(self) -> None:
        splits = get_split_records()
        baseline = ConstantBaselineModel()
        baseline.fit(splits["train"])

        # Check explicit naive labels
        self.assertTrue(baseline.is_naive)
        self.assertEqual(baseline.model_type, "constant_mean_baseline")

        # Mean of targets [1, 3, 5, 7] = 4.0
        self.assertEqual(baseline.mean_target_, 4.0)
        self.assertEqual(baseline.step_count, 1)

        val_preds = baseline.predict(splits["validation"])
        self.assertEqual(val_preds, [4.0])


class TestValidationAndNegativeRegression(unittest.TestCase):
    """Test schema validations and negative regression checks."""

    def test_boolean_features_rejected(self) -> None:
        model = CPURegressionSmokeModel()
        with self.assertRaises(SchemaValidationError):
            model.fit([{"record_id": "r1", "x1": True, "x2": 1, "target": 1}])

    def test_predict_before_fit_raises_runtime_error(self) -> None:
        model = CPURegressionSmokeModel()
        with self.assertRaises(RuntimeError):
            model.predict([{"record_id": "r1", "x1": 0, "x2": 1}])

    def test_negative_regression_fails_when_expected_altered(self) -> None:
        """Verify that altering an expected metric causes assertion failure."""
        splits = get_split_records()
        result = run_regression_smoke_pipeline(
            splits["train"], splits["validation"], splits["test"], seed=42
        )
        actual_step_count = result["step_count"]
        deliberate_wrong_step_count = 999
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_step_count, deliberate_wrong_step_count)


if __name__ == "__main__":
    unittest.main()
