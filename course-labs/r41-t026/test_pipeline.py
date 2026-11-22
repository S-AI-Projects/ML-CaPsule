"""Unit tests for R41-T026 (Building energy efficiency – regression smoke test).

Task: R41-T026 · 6 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

Run offline (CPU-only, Python 3 standard library only):
    python3 -m unittest discover -s course-labs/r41-t026 -p "test_*.py" -v

All tests use the bounded six-row synthetic deterministic formula fixture
(target = 2 * x1 + 1) for pipeline verification only — NOT observations from
the UCI Energy Efficiency dataset and NOT evidence of real-world or clinical
performance.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pipeline  # noqa: E402

# ── Exact analytical expectations for seed=42, lr=0.05, 1 epoch (4 steps) ───

EXPECTED_SGD_STEP_COUNT = 4
EXPECTED_SGD_EPOCHS = 1
EXPECTED_SGD_W1 = 2.02
EXPECTED_SGD_W2 = 0.946
EXPECTED_SGD_STEP_LOSSES = (25.0, 2.25, 8.41, 0.0036)
EXPECTED_SGD_TRAIN_PREDS = (0.946, 2.966, 4.986, 7.006)
EXPECTED_SGD_VAL_PREDS = (9.026,)
EXPECTED_SGD_TEST_PREDS = (11.046,)
EXPECTED_SGD_TRAIN_MSE = 0.001076
EXPECTED_SGD_VAL_MSE = 0.000676
EXPECTED_SGD_TEST_MSE = 0.002116

EXPECTED_NAIVE_MEAN = 4.0
EXPECTED_NAIVE_TRAIN_MSE = 5.0
EXPECTED_NAIVE_VAL_MSE = 25.0
EXPECTED_NAIVE_TEST_MSE = 49.0


class TestDeterministicFormulaFixture(unittest.TestCase):
    """Verify the literal six-row deterministic formula fixture and schema."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.schema_doc = pipeline.load_schema()
        self.records = self.fixture_doc["records"]

    def test_explicit_synthetic_and_formula_labelling(self) -> None:
        self.assertTrue(self.fixture_doc["is_synthetic"])
        self.assertEqual(
            self.fixture_doc["fixture_kind"],
            "deterministic_formula_fixture",
        )
        self.assertIn("deterministic formula fixture", self.fixture_doc["disclaimer"].lower())
        self.assertIn("not observations", self.fixture_doc["disclaimer"].lower())
        self.assertTrue(self.schema_doc["synthetic_only"])

    def test_literal_six_rows_and_formula_values(self) -> None:
        self.assertEqual(len(self.records), 6)
        self.assertLessEqual(len(self.records), pipeline.MAX_SAMPLES)
        self.assertEqual([r["record_id"] for r in self.records], ["r1", "r2", "r3", "r4", "r5", "r6"])
        self.assertEqual([r["x1"] for r in self.records], [0, 1, 2, 3, 4, 5])
        self.assertEqual([r["x2"] for r in self.records], [1, 1, 1, 1, 1, 1])
        self.assertEqual([r["target"] for r in self.records], [1, 3, 5, 7, 9, 11])
        for rec in self.records:
            self.assertEqual(rec["target"], 2 * rec["x1"] + 1)

    def test_schema_declares_required_fields_and_smoke_bounds(self) -> None:
        self.assertEqual(
            tuple(self.schema_doc["required_fields"]),
            pipeline.REQUIRED_RECORD_FIELDS,
        )
        bounds = self.schema_doc["smoke_bounds"]
        self.assertEqual(bounds["max_samples"], 32)
        self.assertEqual(bounds["max_epochs"], 1)
        self.assertEqual(bounds["max_update_steps"], 20)
        self.assertEqual(bounds["fixed_seed"], 42)


class TestExplicitTrainValTestSeparation(unittest.TestCase):
    """Verify explicit disjoint train/validation/test partition assignments."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(
            self.fixture_doc["records"]
        )

    def test_exact_split_record_ids(self) -> None:
        self.assertEqual([r["record_id"] for r in self.partitions.train], ["r1", "r2", "r3", "r4"])
        self.assertEqual([r["record_id"] for r in self.partitions.validation], ["r5"])
        self.assertEqual([r["record_id"] for r in self.partitions.test], ["r6"])

    def test_partitions_are_pairwise_disjoint_and_complete(self) -> None:
        train_ids = {r["record_id"] for r in self.partitions.train}
        val_ids = {r["record_id"] for r in self.partitions.validation}
        test_ids = {r["record_id"] for r in self.partitions.test}
        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(val_ids & test_ids, set())
        self.assertEqual(train_ids | val_ids | test_ids, {"r1", "r2", "r3", "r4", "r5", "r6"})


class TestBoundedSGDSmokeModel(unittest.TestCase):
    """Verify bounded 1-epoch / <=20-step CPU smoke model interface & outputs."""

    def setUp(self) -> None:
        fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(fixture_doc["records"])
        self.model = pipeline.BoundedLinearSGDSmokeRegressor(
            learning_rate=pipeline.DEFAULT_LEARNING_RATE,
            epochs=pipeline.MAX_EPOCHS,
            max_steps=pipeline.MAX_UPDATE_STEPS,
            seed=pipeline.FIXED_SEED,
        ).fit(self.partitions.train)
        self.train_eval = self.model.evaluate("train", self.partitions.train)
        self.val_eval = self.model.evaluate("validation", self.partitions.validation)
        self.test_eval = self.model.evaluate("test", self.partitions.test)

    def test_bounded_step_and_epoch_counts_recorded(self) -> None:
        self.assertEqual(self.model.epochs_completed_, EXPECTED_SGD_EPOCHS)
        self.assertLessEqual(self.model.epochs_completed_, pipeline.MAX_EPOCHS)
        self.assertEqual(self.model.step_count_, EXPECTED_SGD_STEP_COUNT)
        self.assertLessEqual(self.model.step_count_, pipeline.MAX_UPDATE_STEPS)
        self.assertEqual(len(self.model.step_losses_), EXPECTED_SGD_STEP_COUNT)
        self.assertTrue(self.model.asserts_interface_only)
        self.assertFalse(self.model.is_naive_baseline)

    def test_finite_step_losses_and_exact_learned_weights(self) -> None:
        for actual_loss, expected_loss in zip(self.model.step_losses_, EXPECTED_SGD_STEP_LOSSES):
            self.assertTrue(math.isfinite(actual_loss))
            self.assertAlmostEqual(actual_loss, expected_loss, places=12)
        self.assertAlmostEqual(self.model.w1_, EXPECTED_SGD_W1, places=12)
        self.assertAlmostEqual(self.model.w2_, EXPECTED_SGD_W2, places=12)

    def test_prediction_shapes_dtypes_and_exact_values(self) -> None:
        self.assertEqual(self.train_eval.predictions.shape, (4,))
        self.assertEqual(self.val_eval.predictions.shape, (1,))
        self.assertEqual(self.test_eval.predictions.shape, (1,))

        for ev, expected_preds, expected_mse in (
            (self.train_eval, EXPECTED_SGD_TRAIN_PREDS, EXPECTED_SGD_TRAIN_MSE),
            (self.val_eval, EXPECTED_SGD_VAL_PREDS, EXPECTED_SGD_VAL_MSE),
            (self.test_eval, EXPECTED_SGD_TEST_PREDS, EXPECTED_SGD_TEST_MSE),
        ):
            self.assertEqual(ev.predictions.dtype, "float64")
            self.assertTrue(math.isfinite(ev.mse_loss))
            self.assertTrue(math.isfinite(ev.mae_loss))
            self.assertAlmostEqual(ev.mse_loss, expected_mse, places=12)
            for actual_p, expected_p in zip(ev.predictions.values, expected_preds):
                self.assertIs(type(actual_p), float)
                self.assertTrue(math.isfinite(actual_p))
                self.assertAlmostEqual(actual_p, expected_p, places=12)

    def test_max_steps_cap_is_enforced_within_epoch(self) -> None:
        capped = pipeline.BoundedLinearSGDSmokeRegressor(
            learning_rate=0.05,
            epochs=1,
            max_steps=2,
            seed=42,
        ).fit(self.partitions.train)
        self.assertEqual(capped.step_count_, 2)
        self.assertEqual(len(capped.step_losses_), 2)

    def test_deterministic_reproducibility_with_fixed_seed(self) -> None:
        run_a = pipeline.run_smoke_pipeline()
        run_b = pipeline.run_smoke_pipeline()
        self.assertEqual(run_a, run_b)
        self.assertTrue(run_a["verification_only"])


class TestExplicitConstantNaiveBaseline(unittest.TestCase):
    """Verify the explicitly labelled constant/naive training-mean baseline."""

    def setUp(self) -> None:
        fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(fixture_doc["records"])
        self.baseline = pipeline.NaiveMeanBaseline(seed=pipeline.FIXED_SEED).fit(
            self.partitions.train
        )

    def test_naive_baseline_explicit_labelling_and_train_only_mean(self) -> None:
        self.assertTrue(self.baseline.is_naive_baseline)
        self.assertEqual(self.baseline.model_kind, "constant_naive_baseline")
        self.assertEqual(self.baseline.step_count_, 1)
        self.assertEqual(self.baseline.epochs_completed_, 1)
        # Train targets are [1, 3, 5, 7] -> mean 4.0 (all-6 mean would be 6.0)
        self.assertAlmostEqual(self.baseline.train_mean_, EXPECTED_NAIVE_MEAN, places=12)
        self.assertNotAlmostEqual(self.baseline.train_mean_, 6.0, places=6)

    def test_naive_baseline_shapes_and_losses(self) -> None:
        tr = self.baseline.evaluate("train", self.partitions.train)
        va = self.baseline.evaluate("validation", self.partitions.validation)
        te = self.baseline.evaluate("test", self.partitions.test)

        self.assertEqual(tr.predictions.shape, (4,))
        self.assertEqual(va.predictions.shape, (1,))
        self.assertEqual(te.predictions.shape, (1,))
        self.assertEqual(tr.predictions.values, (4.0, 4.0, 4.0, 4.0))
        self.assertEqual(va.predictions.values, (4.0,))
        self.assertEqual(te.predictions.values, (4.0,))
        self.assertAlmostEqual(tr.mse_loss, EXPECTED_NAIVE_TRAIN_MSE, places=12)
        self.assertAlmostEqual(va.mse_loss, EXPECTED_NAIVE_VAL_MSE, places=12)
        self.assertAlmostEqual(te.mse_loss, EXPECTED_NAIVE_TEST_MSE, places=12)


class TestSmokeBoundsAndFailureCases(unittest.TestCase):
    """Verify smoke bounds, contamination guards, and invalid-input failures."""

    def setUp(self) -> None:
        fixture_doc = pipeline.load_fixture()
        self.records = fixture_doc["records"]
        self.partitions = pipeline.validate_and_partition_fixture(self.records)

    def test_exceeding_32_sample_bound_is_rejected(self) -> None:
        oversized = [
            {"record_id": f"r{i}", "x1": i, "x2": 1, "target": 2 * i + 1, "split": "train"}
            for i in range(33)
        ]
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor().fit(oversized)
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.validate_and_partition_fixture(oversized)

    def test_exceeding_epoch_or_step_or_lr_bounds_is_rejected(self) -> None:
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor(epochs=2)
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor(epochs=0)
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor(max_steps=21)
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor(max_steps=0)
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor(learning_rate=0.0)
        with self.assertRaises(pipeline.SmokeBoundError):
            pipeline.BoundedLinearSGDSmokeRegressor(learning_rate=float("nan"))

    def test_non_train_row_in_fit_is_rejected(self) -> None:
        leaked_train = list(self.partitions.train) + [dict(self.partitions.validation[0])]
        with self.assertRaises(pipeline.SplitContaminationError):
            pipeline.BoundedLinearSGDSmokeRegressor().fit(leaked_train)
        with self.assertRaises(pipeline.SplitContaminationError):
            pipeline.NaiveMeanBaseline().fit(leaked_train)

    def test_overlapping_record_id_across_splits_rejected(self) -> None:
        contaminated = copy.deepcopy(self.records)
        contaminated.append(
            {"record_id": "r1", "x1": 6, "x2": 1, "target": 13, "split": "test"}
        )
        with self.assertRaises(pipeline.SplitContaminationError):
            pipeline.validate_and_partition_fixture(contaminated)

    def test_train_row_passed_into_held_out_evaluate_rejected(self) -> None:
        model = pipeline.BoundedLinearSGDSmokeRegressor().fit(self.partitions.train)
        with self.assertRaises(pipeline.SplitContaminationError):
            model.evaluate("test", [dict(self.partitions.train[0])])

    def test_predict_before_fit_raises(self) -> None:
        unfitted_sgd = pipeline.BoundedLinearSGDSmokeRegressor()
        with self.assertRaises(pipeline.ModelNotFittedError):
            unfitted_sgd.predict(self.partitions.validation)

        unfitted_naive = pipeline.NaiveMeanBaseline()
        with self.assertRaises(pipeline.ModelNotFittedError):
            unfitted_naive.predict(self.partitions.test)

    def test_non_finite_missing_or_bool_values_rejected(self) -> None:
        for bad_val in (None, float("nan"), float("inf"), True, "bad"):
            bad_rows = copy.deepcopy(list(self.partitions.train))
            bad_rows[0]["x1"] = bad_val
            with self.assertRaises(pipeline.SchemaValidationError):
                pipeline.BoundedLinearSGDSmokeRegressor().fit(bad_rows)

        bad_formula = copy.deepcopy(self.records)
        bad_formula[0]["target"] = 999
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.validate_and_partition_fixture(bad_formula, verify_formula=True)


class TestDeliberateMutationGuard(unittest.TestCase):
    """Prove assertions fail when an expected smoke-run output is altered."""

    def test_altered_expected_prediction_fails_assertion(self) -> None:
        report = pipeline.run_smoke_pipeline()
        actual_test_pred = report["models"]["bounded_sgd_smoke"]["splits"]["test"]["predictions"][0]
        deliberately_wrong = 999.0
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(actual_test_pred, deliberately_wrong, places=6)


if __name__ == "__main__":
    unittest.main()
