"""Unit tests for bounded CPU regression pipeline smoke test (Task R41-T016)."""

import math
import unittest
from pipeline import BoundedCPULinearRegression, load_fixture, run_pipeline


class TestBoundedCPUSmokeRun(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.metadata = self.payload["metadata"]
        self.records = self.payload["records"]
        self.by_id = {r["record_id"]: r for r in self.records}

    def test_fixture_deterministic_formula_and_bounds(self) -> None:
        # Fixture size must be <= 32 samples (here exactly 6)
        self.assertLessEqual(len(self.records), 32)
        self.assertEqual(len(self.records), 6)

        # Assert deterministic formula: target = 2 * x1 + 1
        for r in self.records:
            expected_target = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(r["target"], expected_target, places=4)
            self.assertEqual(r["x2"], 1.0)

    def test_smoke_run_outputs_shapes_and_step_count(self) -> None:
        result = run_pipeline()

        # Step count bounded to at most 20 update steps
        self.assertLessEqual(result["step_count"], 20)
        self.assertEqual(result["step_count"], 4)

        # Output prediction shapes
        self.assertEqual(result["train_shape"], [4, 1])
        self.assertEqual(result["validation_shape"], [1, 1])
        self.assertEqual(result["test_shape"], [1, 1])

        # Loss history and weights must be finite
        for loss in result["loss_history"]:
            self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isfinite(result["weights"]["w1"]))
        self.assertTrue(math.isfinite(result["weights"]["w2"]))

        # Metric values must be non-negative and finite
        for k, mse in result["metrics"].items():
            self.assertTrue(math.isfinite(mse))
            self.assertGreaterEqual(mse, 0.0)

    def test_unfitted_model_raises_runtime_error(self) -> None:
        model = BoundedCPULinearRegression()
        with self.assertRaisesRegex(RuntimeError, "must be trained before predicting"):
            model.predict([[0.0, 1.0]])

    def test_mismatched_inputs_raise_value_error(self) -> None:
        model = BoundedCPULinearRegression()
        with self.assertRaisesRegex(ValueError, "Mismatch between features and targets"):
            model.fit([[0.0, 1.0], [1.0, 1.0]], [1.0])


if __name__ == "__main__":
    unittest.main()
